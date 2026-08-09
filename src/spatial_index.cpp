#include <pybind11/stl.h>
#include <s2/mutable_s2shape_index.h>
#include <s2/s1angle.h>
#include <s2/s1chord_angle.h>
#include <s2/s2cell_id.h>
#include <s2/s2cell_union.h>
#include <s2/s2closest_edge_query.h>
#include <s2/s2region_coverer.h>
#include <s2geography.h>
#include <s2geography/index.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "constants.hpp"
#include "geography.hpp"
#include "predicates.hpp"
#include "pybind11.hpp"
#include "s2_tmp_memory_budget.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

namespace {

/*
** Sets s2geometry's process-wide index-build temporary memory budget
** (``FLAGS_s2shape_index_tmp_memory_budget``) for the lifetime of the guard and
** puts the previous value back when it goes out of scope, so an override cannot
** leak out of the build it was requested for.
**
** The flag is read and written through spherely::get/set_s2_tmp_memory_budget,
** which own the declaration it needs (see s2_tmp_memory_budget.cpp) -- reaching
** it directly from this file would not link against a shared s2 on Windows.
*/
class TmpMemoryBudgetGuard {
public:
    explicit TmpMemoryBudgetGuard(std::int64_t bytes) : m_previous(get_s2_tmp_memory_budget()) {
        set_s2_tmp_memory_budget(bytes);
    }

    ~TmpMemoryBudgetGuard() {
        set_s2_tmp_memory_budget(m_previous);
    }

    TmpMemoryBudgetGuard(const TmpMemoryBudgetGuard&) = delete;
    TmpMemoryBudgetGuard& operator=(const TmpMemoryBudgetGuard&) = delete;

private:
    std::int64_t m_previous;
};

}  // namespace

/*
** A spatial index over a collection of Geography objects, backed by
** s2geography::GeographyIndex (a MutableS2ShapeIndex). Provides fast candidate
** lookup similar to shapely's STRtree.
*/
class SpatialIndex {
public:
    SpatialIndex(const py::array_t<PyObjectGeography>& geographies) {
        if (geographies.ndim() != 1) {
            throw py::type_error("geographies must be a 1-dimensional array");
        }

        // store a shallow copy of the input array so that replacing its
        // elements afterwards cannot drop Geography objects whose S2Shapes
        // are still borrowed by the index
        m_geographies = py::array_t<PyObjectGeography>::ensure(geographies.attr("copy")());
        m_index = std::make_unique<s2geog::GeographyIndex>();

        auto n = m_geographies.size();
        auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
        for (py::ssize_t i = 0; i < n; i++) {
            m_index->Add(data[i].as_geog_ptr()->geog(), static_cast<int>(i));
        }
    }

    std::size_t size() const {
        return static_cast<std::size_t>(m_geographies.size());
    }

    // True once the queued updates have been applied (no build work pending).
    bool is_built() const {
        return m_index->ShapeIndex().is_fresh();
    }

    // Apply the updates queued by the constructor, optionally under a raised
    // s2 temporary memory budget. A no-op once the index is built.
    void build(std::optional<std::int64_t> tmp_memory_budget) const {
        if (tmp_memory_budget.has_value() && *tmp_memory_budget <= 0) {
            throw py::value_error("tmp_memory_budget must be a positive number of bytes, got " +
                                  std::to_string(*tmp_memory_budget) +
                                  " (s2geometry's default budget is 104857600 bytes)");
        }

        // nothing pending: no build to tune, so leave the process-wide flag alone
        if (is_built()) {
            return;
        }

        const auto& shape_index = m_index->ShapeIndex();
        if (tmp_memory_budget.has_value()) {
            TmpMemoryBudgetGuard guard(*tmp_memory_budget);
            shape_index.ForceBuild();
        } else {
            shape_index.ForceBuild();
        }
    }

    py::array geometries() const {
        return m_geographies;
    }

    // Scalar query: single Geography -> 1-d array of sorted tree indices.
    py::array_t<py::ssize_t> query(const Geography& geography,
                                   std::optional<std::string> predicate) const {
        auto pred = make_predicate(predicate);
        auto results = query_one(geography, pred ? &*pred : nullptr);
        return to_index_array(results);
    }

    // Array query: 1-d array of Geography -> (2, K) array of
    // (input index, tree index) pairs.
    py::array_t<py::ssize_t> query(const py::array_t<PyObjectGeography>& geographies,
                                   std::optional<std::string> predicate) const {
        if (geographies.ndim() != 1) {
            throw py::type_error(
                "query geography must be a Geography or a 1-dimensional array of Geography");
        }

        auto pred = make_predicate(predicate);

        auto n = geographies.size();
        auto* data = static_cast<PyObjectGeography*>(geographies.request().ptr);
        std::vector<py::ssize_t> input_idx;
        std::vector<py::ssize_t> tree_idx;
        for (py::ssize_t i = 0; i < n; i++) {
            auto results = query_one(*data[i].as_geog_ptr(), pred ? &*pred : nullptr);
            for (int t : results) {
                input_idx.push_back(i);
                tree_idx.push_back(static_cast<py::ssize_t>(t));
            }
        }

        return to_pairs_array(input_idx, tree_idx);
    }

    // Scalar nearest query: single Geography -> 1-d array of sorted tree
    // indices of the nearest geographies (optionally with their distance).
    py::object query_nearest(const Geography& geography,
                             std::optional<double> max_distance,
                             bool return_distance,
                             bool exclusive,
                             bool all_matches,
                             double radius) const {
        S1ChordAngle distance;
        auto results = query_nearest_one(
            geography, to_chord_angle(max_distance, radius), exclusive, all_matches, &distance);
        auto indices = to_index_array(results);
        if (!return_distance) {
            return indices;
        }

        auto n = results.size();
        py::array_t<double> distances(static_cast<py::ssize_t>(n));
        auto dist_data = distances.mutable_unchecked<1>();
        auto dist = distance.ToAngle().radians() * radius;
        for (std::size_t i = 0; i < n; i++) {
            dist_data(static_cast<py::ssize_t>(i)) = dist;
        }
        return py::make_tuple(indices, distances);
    }

    // Array nearest query: 1-d array of Geography -> (2, K) array of
    // (input index, tree index) pairs (optionally with their distances).
    py::object query_nearest(const py::array_t<PyObjectGeography>& geographies,
                             std::optional<double> max_distance,
                             bool return_distance,
                             bool exclusive,
                             bool all_matches,
                             double radius) const {
        if (geographies.ndim() != 1) {
            throw py::type_error(
                "query geography must be a Geography or a 1-dimensional array of Geography");
        }

        auto max_dist = to_chord_angle(max_distance, radius);
        auto n = geographies.size();
        auto* data = static_cast<PyObjectGeography*>(geographies.request().ptr);
        std::vector<py::ssize_t> input_idx;
        std::vector<py::ssize_t> tree_idx;
        std::vector<double> dists;
        for (py::ssize_t i = 0; i < n; i++) {
            S1ChordAngle distance;
            auto results = query_nearest_one(
                *data[i].as_geog_ptr(), max_dist, exclusive, all_matches, &distance);
            for (int t : results) {
                input_idx.push_back(i);
                tree_idx.push_back(static_cast<py::ssize_t>(t));
                dists.push_back(distance.ToAngle().radians() * radius);
            }
        }

        auto pairs = to_pairs_array(input_idx, tree_idx);
        if (!return_distance) {
            return pairs;
        }

        auto k = dists.size();
        py::array_t<double> distances(static_cast<py::ssize_t>(k));
        auto dist_data = distances.mutable_unchecked<1>();
        for (std::size_t j = 0; j < k; j++) {
            dist_data(static_cast<py::ssize_t>(j)) = dists[j];
        }
        return py::make_tuple(pairs, distances);
    }

private:
    std::unique_ptr<s2geog::GeographyIndex> m_index;
    // also keeps the Geography objects alive (the index borrows their S2Shapes)
    py::array_t<PyObjectGeography> m_geographies;

    static std::optional<PredicateFunc> make_predicate(const std::optional<std::string>& name) {
        if (!name.has_value()) {
            return std::nullopt;
        }
        return get_predicate(*name);
    }

    // Return the sorted tree indices whose cells overlap the query geography,
    // optionally refined by ``pred`` (predicate(query, candidate)).
    std::vector<int> query_one(const Geography& query_geog, const PredicateFunc* pred) const {
        std::unordered_set<int> hits;

        auto region = query_geog.geog().Region();
        S2RegionCoverer coverer;
        std::vector<S2CellId> covering;
        coverer.GetCovering(*region, &covering);

        s2geog::GeographyIndex::Iterator iter(m_index.get());
        iter.Query(covering, &hits);

        std::vector<int> results;
        if (pred == nullptr) {
            results.assign(hits.begin(), hits.end());
        } else {
            const auto& query_index = query_geog.geog_index();
            auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
            for (int candidate : hits) {
                auto* cand_geog = data[candidate].as_geog_ptr();
                if ((*pred)(query_index, cand_geog->geog_index())) {
                    results.push_back(candidate);
                }
            }
        }

        std::sort(results.begin(), results.end());
        return results;
    }

    // Convert a distance in meters (given the radius) to a chord angle.
    static std::optional<S1ChordAngle> to_chord_angle(std::optional<double> distance,
                                                      double radius) {
        // the radius scales both the returned distances and ``max_distance``,
        // so it has to be checked even when there is no distance bound
        if (!std::isfinite(radius) || radius <= 0) {
            throw py::value_error("radius must be a finite value greater than 0");
        }
        if (!distance.has_value()) {
            return std::nullopt;
        }
        // NaN would compare false here and silently mean "unbounded"
        if (!std::isfinite(*distance) || *distance <= 0) {
            throw py::value_error("max_distance must be a finite value greater than 0");
        }
        auto bound = S1ChordAngle(S1Angle::Radians(*distance / radius));
        // the meters -> radians -> chord angle conversion here and the chord
        // angle -> radians -> meters conversion of the returned distances do
        // not round trip exactly (they disagree by a couple of ULPs), so a
        // distance reported by ``query_nearest`` would otherwise be rejected
        // as a max_distance about a quarter of the time. Widen the bound by a
        // few ULPs to keep it inclusive (S1ChordAngle::set_inclusive_max_distance
        // only bumps it by one). FromLength2 clamps to a straight angle.
        return S1ChordAngle::FromLength2(bound.length2() *
                                         (1 + 8 * std::numeric_limits<double>::epsilon()));
    }

    // Return the sorted tree indices of every geography whose distance to the
    // target is exactly ``dist`` (``dist`` must be the minimum distance over
    // the geographies not skipped, so that no geography has an edge in
    // ``(0, dist)``).
    std::vector<int> collect_at_distance(S2ClosestEdgeQuery& query,
                                         S2ClosestEdgeQuery::ShapeIndexTarget& target,
                                         S1ChordAngle dist) const {
        query.mutable_options()->set_max_results(S2ClosestEdgeQuery::Options::kMaxMaxResults);
        query.mutable_options()->set_inclusive_max_distance(dist);
        std::unordered_set<int> hits;
        for (const auto& result : query.FindClosestEdges(&target)) {
            if (result.distance() == dist) {
                hits.insert(m_index->value(result.shape_id()));
            }
        }
        std::vector<int> results(hits.begin(), hits.end());
        std::sort(results.begin(), results.end());
        return results;
    }

    // ``collect_at_distance``, reduced to the lowest tree index if
    // ``all_matches`` is false: which tie is returned is unspecified by the
    // API but is picked deterministically (the results are sorted).
    std::vector<int> nearest_at_distance(S2ClosestEdgeQuery& query,
                                         S2ClosestEdgeQuery::ShapeIndexTarget& target,
                                         S1ChordAngle dist,
                                         bool all_matches) const {
        auto results = collect_at_distance(query, target, dist);
        if (!all_matches && results.size() > 1) {
            results.resize(1);
        }
        return results;
    }

    // Return the sorted tree indices of the geographies nearest to
    // ``query_geog`` (every tie at the minimum distance if ``all_matches``,
    // otherwise the lowest of them), storing the minimum distance in
    // ``*distance`` (Infinity if there is no result). Geographies farther
    // than ``max_distance`` (inclusive) are never returned; if ``exclusive``,
    // neither are geographies equal to ``query_geog``.
    std::vector<int> query_nearest_one(const Geography& query_geog,
                                       std::optional<S1ChordAngle> max_distance,
                                       bool exclusive,
                                       bool all_matches,
                                       S1ChordAngle* distance) const {
        *distance = S1ChordAngle::Infinity();
        if (query_geog.is_empty()) {
            return {};
        }

        S2ClosestEdgeQuery query(&m_index->ShapeIndex());
        // count the interior of indexed polygons (resp. of the query
        // geography) as distance zero, consistent with spherely.distance
        query.mutable_options()->set_include_interiors(true);
        if (max_distance.has_value()) {
            query.mutable_options()->set_inclusive_max_distance(*max_distance);
        }
        S2ClosestEdgeQuery::ShapeIndexTarget target(&query_geog.geog_index().ShapeIndex());
        target.set_include_interiors(true);

        auto closest = query.FindClosestEdge(&target);
        if (closest.is_empty()) {
            return {};
        }
        auto min_dist = closest.distance();

        if (!exclusive || min_dist > S1ChordAngle::Zero()) {
            // geographies equal to the query would be at distance zero, so
            // nothing needs to be excluded here
            *distance = min_dist;
            return nearest_at_distance(query, target, min_dist, all_matches);
        }

        // exclusive with matches at distance zero: split those into
        // geographies equal to the query (excluded) and others (returned)
        auto zero_hits = collect_at_distance(query, target, S1ChordAngle::Zero());
        auto equals_pred = get_predicate("equals");
        const auto& query_index = query_geog.geog_index();
        auto* data = static_cast<PyObjectGeography*>(m_geographies.request().ptr);
        std::vector<int> non_equal;
        for (int t : zero_hits) {
            if (!equals_pred(query_index, data[t].as_geog_ptr()->geog_index())) {
                non_equal.push_back(t);
            }
        }
        if (!non_equal.empty()) {
            *distance = S1ChordAngle::Zero();
            if (!all_matches) {
                return {non_equal.front()};  // sorted: the lowest tree index
            }
            return non_equal;
        }

        // every geography at distance zero equals the query: look for the
        // nearest one at a distance > 0. Every edge of an equal geography is
        // at distance zero (equal geographies cover each other), so the first
        // result at a positive distance -- results are sorted by distance --
        // gives the minimum distance over the non-equal geographies.
        if (max_distance.has_value()) {
            query.mutable_options()->set_inclusive_max_distance(*max_distance);
        } else {
            query.mutable_options()->set_max_distance(S1ChordAngle::Infinity());
        }
        constexpr int kMaxResults = S2ClosestEdgeQuery::Options::kMaxMaxResults;
        for (int max_results = 16;;) {
            query.mutable_options()->set_max_results(max_results);
            auto results = query.FindClosestEdges(&target);
            for (const auto& result : results) {
                if (result.distance() > S1ChordAngle::Zero()) {
                    *distance = result.distance();
                    return nearest_at_distance(query, target, result.distance(), all_matches);
                }
            }
            if (results.size() < static_cast<std::size_t>(max_results) ||
                max_results == kMaxResults) {
                // all edges within max_distance seen, none at distance > 0
                return {};
            }
            // saturate instead of overflowing (which would be undefined)
            max_results = max_results > kMaxResults / 2 ? kMaxResults : max_results * 2;
        }
    }

    static py::array_t<py::ssize_t> to_index_array(const std::vector<int>& indices) {
        auto n = indices.size();
        py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(n));
        auto out_data = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < n; i++) {
            out_data(static_cast<py::ssize_t>(i)) = static_cast<py::ssize_t>(indices[i]);
        }
        return out;
    }

    static py::array_t<py::ssize_t> to_pairs_array(const std::vector<py::ssize_t>& input_idx,
                                                   const std::vector<py::ssize_t>& tree_idx) {
        auto k = input_idx.size();
        py::array_t<py::ssize_t> out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(k)});
        auto out_data = out.mutable_unchecked<2>();
        for (std::size_t j = 0; j < k; j++) {
            out_data(0, static_cast<py::ssize_t>(j)) = input_idx[j];
            out_data(1, static_cast<py::ssize_t>(j)) = tree_idx[j];
        }
        return out;
    }
};

void init_spatial_index(py::module& m) {
    py::class_<SpatialIndex>(m, "SpatialIndex", R"pbdoc(
        A spatial index for a collection of geographies.

        The index allows fast retrieval of the geographies that may interact
        with a given geography, e.g., for accelerating spatial joins. It is
        built on top of s2geometry's shape index and is conceptually similar to
        shapely's ``STRtree``.

        The underlying shape index is built lazily: the constructor only queues
        the geographies and the first query pays for the build. Use
        :py:meth:`SpatialIndex.build` to do that work up front instead.

        Parameters
        ----------
        geographies : array_like
            An array (or other sequence) of :py:class:`Geography` objects. Empty
            geographies are indexed but never returned by queries.

    )pbdoc")
        .def(py::init<const py::array_t<PyObjectGeography>&>(),
             py::arg("geographies"),
             "__init__(self, geographies)")
        .def("build",
             &SpatialIndex::build,
             py::arg("tmp_memory_budget") = py::none(),
             R"pbdoc(build(tmp_memory_budget=None)

        Force the deferred index build, optionally under a raised temporary
        memory budget.

        The index is built lazily, so without this call the first query pays
        the whole build cost. Calling it is optional, never changes the result
        of a query made with a ``predicate``, and does nothing once the index
        is built (see :py:attr:`SpatialIndex.is_built`).

        Parameters
        ----------
        tmp_memory_budget : int, optional
            Temporary memory budget for this build, in bytes. s2geometry builds
            the index in batches sized to fit it, and many batches cost more
            than one, so the knob is only useful *raised*; about 226 bytes per
            edge is enough to build in a single batch. Must be strictly
            positive, and must fit in a signed 64-bit integer (larger values
            raise ``TypeError``). The previous value is put back when the call
            returns, so the override does not leak into any later build.

            ``None`` does *not* mean "no budget": it leaves s2geometry's
            setting untouched (104857600 bytes, i.e. 100 MB, unless something
            else in the process changed it), so the build still batches.

        Notes
        -----
        Peak resident memory can rise by roughly the budget granted, and
        nothing is granted unless a budget is passed.

        The budget changes where s2geometry places its batch boundaries, and so
        how it subdivides the index. A query made without a ``predicate`` can
        therefore return a different candidate set under a different budget --
        candidates may be added *or* dropped, and the sets for two budgets need
        not be nested. What does not change is that the candidate set is always
        a superset of the true matches, so any ``predicate`` refines it to the
        same answer.

        This call holds the GIL for its whole duration -- minutes, for a large
        collection -- and the budget is a process-wide s2geometry setting for
        that duration.

        Examples
        --------
        >>> import spherely
        >>> index = spherely.SpatialIndex(geographies)
        >>> index.build(tmp_memory_budget=4 * 1024**3)  # 4 GiB

    )pbdoc")
        .def("__len__", &SpatialIndex::size)
        .def_property_readonly("is_built",
                               &SpatialIndex::is_built,
                               R"pbdoc(Whether the underlying shape index has been built.

        ``False`` between the constructor and the first
        :py:meth:`SpatialIndex.build` or query, ``True`` afterwards. An index
        over an empty collection is ``True`` from the start.

    )pbdoc")
        .def_property_readonly("geometries",
                               &SpatialIndex::geometries,
                               "The array of geographies in the index (in input order).")
        .def("query",
             py::overload_cast<const Geography&, std::optional<std::string>>(&SpatialIndex::query,
                                                                             py::const_),
             py::arg("geography"),
             py::arg("predicate") = py::none(),
             R"pbdoc(query(geography, predicate=None)

        Return the integer indices of all geographies in the index whose cells
        overlap the given geography (a coarse candidate set), optionally refined
        by a spatial predicate.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            The geography or geographies to query.
        predicate : str, optional
            If provided, only return candidates for which
            ``predicate(geography, indexed_geography)`` is True. One of
            "intersects", "within", "contains", "covers", "covered_by",
            "touches" or "equals".

        Returns
        -------
        ndarray
            If ``geography`` is a scalar, a 1-d array of (sorted) indices into
            the index. If ``geography`` is an array, a ``(2, N)`` array where the
            first row holds the input geography indices and the second row the
            matching index (tree) indices.

    )pbdoc")
        .def("query",
             py::overload_cast<const py::array_t<PyObjectGeography>&, std::optional<std::string>>(
                 &SpatialIndex::query, py::const_),
             py::arg("geography"),
             py::arg("predicate") = py::none())
        .def(
            "query_nearest",
            py::overload_cast<const Geography&, std::optional<double>, bool, bool, bool, double>(
                &SpatialIndex::query_nearest, py::const_),
            py::arg("geography"),
            py::arg("max_distance") = py::none(),
            py::arg("return_distance") = false,
            py::arg("exclusive") = false,
            py::arg("all_matches") = true,
            py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS,
            R"pbdoc(query_nearest(geography, max_distance=None, return_distance=False, exclusive=False, all_matches=True, radius=spherely.EARTH_RADIUS_METERS)

        Return the integer indices of the geographies in the index that are
        nearest to the given geography, similar to shapely's
        ``STRtree.query_nearest``.

        The distance to a geography is zero if the query geography intersects
        it (including polygon interiors), consistent with
        :py:func:`spherely.distance`. Empty geographies (in the index or as
        query) are never matched.

        Parameters
        ----------
        geography : :py:class:`Geography` or array_like
            The geography or geographies to query.
        max_distance : float, optional
            The maximum distance (in meters, inclusive) within which to query
            for the nearest geographies. Must be a finite value greater than 0
            (pass None, the default, for no bound).
        return_distance : bool, default False
            If True, also return the distance(s) to the nearest geographies.
        exclusive : bool, default False
            If True, geographies in the index that are equal to the query
            geography are not returned.
        all_matches : bool, default True
            If True, return all geographies tied at the minimum distance. If
            False, return only one of them: the one with the lowest index
            (shapely leaves the choice arbitrary, spherely makes it
            deterministic).
        radius : float, optional
            Radius of Earth in meters, default 6,371,010. Used to scale the
            returned distances and to interpret ``max_distance``. Must be a
            finite value greater than 0.

        Returns
        -------
        ndarray or tuple of ndarray
            If ``geography`` is a scalar, a 1-d array of (sorted) indices into
            the index. If ``geography`` is an array, a ``(2, K)`` array where
            the first row holds the input geography indices and the second row
            the matching index (tree) indices. If ``return_distance`` is True,
            a ``(indices, distances)`` tuple where ``distances`` holds the
            distance in meters for each match.

    )pbdoc")
        .def("query_nearest",
             py::overload_cast<const py::array_t<PyObjectGeography>&,
                               std::optional<double>,
                               bool,
                               bool,
                               bool,
                               double>(&SpatialIndex::query_nearest, py::const_),
             py::arg("geography"),
             py::arg("max_distance") = py::none(),
             py::arg("return_distance") = false,
             py::arg("exclusive") = false,
             py::arg("all_matches") = true,
             py::arg("radius") = numeric_constants::EARTH_RADIUS_METERS);

    // Not public API: exposed so that the test suite can move the process-wide
    // budget off its default and assert what SpatialIndex.build() puts back.
    m.def("_s2_tmp_memory_budget",
          &get_s2_tmp_memory_budget,
          "Return s2geometry's current index-build temporary memory budget, in bytes.");
    m.def("_set_s2_tmp_memory_budget",
          &set_s2_tmp_memory_budget,
          py::arg("bytes"),
          "Set s2geometry's index-build temporary memory budget, in bytes.");
}
