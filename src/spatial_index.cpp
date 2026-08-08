#include <pybind11/stl.h>
#include <s2/base/commandlineflags.h>
#include <s2/mutable_s2shape_index.h>
#include <s2/s2cell_id.h>
#include <s2/s2cell_union.h>
#include <s2/s2region_coverer.h>
#include <s2geography.h>
#include <s2geography/index.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "geography.hpp"
#include "predicates.hpp"
#include "pybind11.hpp"

namespace py = pybind11;
namespace s2geog = s2geography;
using namespace spherely;

namespace {

/*
** Sets s2geometry's index-build temporary memory budget
** (``FLAGS_s2shape_index_tmp_memory_budget``, declared in
** s2/mutable_s2shape_index.h) for the lifetime of the guard and puts the
** previous value back on destruction, including when the build throws.
**
** The flag is process-wide, so the guard exists to keep the override from
** leaking out of the single build it was requested for. It is read and written
** with absl::GetFlag / absl::SetFlag; s2/base/commandlineflags.h is included
** for those (it pulls in <absl/flags/flag.h>) so that the absl flags
** dependency is expressed through s2's own header, the way s2 does it.
*/
class TmpMemoryBudgetGuard {
public:
    explicit TmpMemoryBudgetGuard(std::int64_t bytes)
        : m_previous(absl::GetFlag(FLAGS_s2shape_index_tmp_memory_budget)) {
        absl::SetFlag(&FLAGS_s2shape_index_tmp_memory_budget, bytes);
    }

    ~TmpMemoryBudgetGuard() {
        absl::SetFlag(&FLAGS_s2shape_index_tmp_memory_budget, m_previous);
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

    // True once the queued updates have been applied, i.e. the index can be
    // queried without doing any build work.
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

        // nothing is pending, so there is no build to tune: return without
        // touching the process-wide budget flag
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

        auto k = input_idx.size();
        py::array_t<py::ssize_t> out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(k)});
        auto out_data = out.mutable_unchecked<2>();
        for (std::size_t j = 0; j < k; j++) {
            out_data(0, static_cast<py::ssize_t>(j)) = input_idx[j];
            out_data(1, static_cast<py::ssize_t>(j)) = tree_idx[j];
        }
        return out;
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

    static py::array_t<py::ssize_t> to_index_array(const std::vector<int>& indices) {
        auto n = indices.size();
        py::array_t<py::ssize_t> out(static_cast<py::ssize_t>(n));
        auto out_data = out.mutable_unchecked<1>();
        for (std::size_t i = 0; i < n; i++) {
            out_data(static_cast<py::ssize_t>(i)) = static_cast<py::ssize_t>(indices[i]);
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
        the geographies, and the first query pays for the build. Use
        :py:meth:`SpatialIndex.build` to do that work up front, and to tune it
        for large collections.

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

        The index is built lazily, so the constructor is cheap and the first
        query pays the whole build cost. This method makes that cost explicit
        (and keeps it out of the timing of the first query), and lets it be
        tuned for large collections.

        s2geometry builds the index in batches sized to fit a temporary memory
        budget of 104857600 bytes (100 MB) by default, and each batch
        re-absorbs the cells built so far. Many batches therefore cost
        noticeably more than one, and collections of millions of edges can
        spend minutes in the first query. Raising the budget so that the index
        is built in fewer (ideally one) batches makes the build close to linear
        in the number of edges.

        Parameters
        ----------
        tmp_memory_budget : int, optional
            Temporary memory budget for this build, **in bytes** -- the unit of
            the underlying s2geometry setting
            (``FLAGS_s2shape_index_tmp_memory_budget``, default 104857600, i.e.
            100 MB). Must be strictly positive, and must fit in a signed 64-bit
            integer (larger values raise ``TypeError``). The previous value is
            put back when the call returns, so the override does not leak into
            any later build. ``None`` does *not* mean "no budget": it leaves
            s2geometry's setting untouched (100 MB unless something else in the
            process changed it), so the build still batches and costs what the
            first query would have. Pass a raised value to avoid that.

            This knob is only useful *raised*. A budget below the one in effect
            splits the build into more batches and makes it slower -- 1 MB
            measured about ten times slower than the default on a 634k-edge
            collection. Mind the unit: ``tmp_memory_budget=100`` is 100
            *bytes*, not 100 MB, and is accepted.

        Notes
        -----
        Memory: the budget is a licence to hold that much scratch at once, so
        peak resident memory can rise by roughly the amount granted. s2geometry
        needs on the order of 140 bytes of temporary memory per edge, so
        ``n_edges * 140`` is a reasonable budget for building in a single
        batch. Nothing is granted unless a budget is passed, so the extra
        memory is always opt-in.

        Calling this method is optional and never changes query results. It is
        idempotent: a second call, or a call after the index has already been
        built by a query, does nothing (and leaves the budget untouched). See
        :py:attr:`SpatialIndex.is_built`.

        Threads: this call holds the GIL for its whole duration, so no other
        Python thread runs until the build finishes -- minutes, for a large
        collection. That is not a regression (a first query does the same build
        under the same GIL), but it is worth knowing before calling it from a
        thread. The budget is a process-wide s2geometry setting for the
        duration of the call, so an index being built concurrently in a
        non-Python thread would see the override too.

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
        over an empty collection has nothing to queue and is ``True`` from the
        start.

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
             py::arg("predicate") = py::none());

    // Not public API: exposed so that the test suite can assert that
    // SpatialIndex.build() puts the process-wide budget back.
    m.def(
        "_s2_tmp_memory_budget",
        []() { return absl::GetFlag(FLAGS_s2shape_index_tmp_memory_budget); },
        "Return s2geometry's current index-build temporary memory budget, in bytes.");
}
