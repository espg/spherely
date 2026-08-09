from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import spherely


def _pairs(result: npt.NDArray[np.intp]) -> set[tuple[Any, ...]]:
    # the (2, N) result of an array query as a set of (input, tree) pairs
    return set(map(tuple, result.T))


@pytest.fixture
def geographies() -> npt.NDArray[Any]:
    # three points near (0, 0)..(2, 2) and one far-away point
    return np.array(
        [
            spherely.create_point(0, 0),
            spherely.create_point(1, 1),
            spherely.create_point(2, 2),
            spherely.create_point(50, 50),
        ]
    )


def test_spatial_index_len(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert len(tree) == 4


def test_spatial_index_geometries(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    geoms = tree.geometries
    assert isinstance(geoms, np.ndarray)
    assert geoms.shape == (4,)
    assert all(spherely.equals(a, b) for a, b in zip(geoms, geographies))


def test_spatial_index_from_list() -> None:
    tree = spherely.SpatialIndex(
        [spherely.create_point(0, 0), spherely.create_point(1, 1)]
    )
    assert len(tree) == 2


def test_spatial_index_copies_input_array() -> None:
    geoms = np.array([spherely.create_point(0, 0), spherely.create_point(1, 1)])
    tree = spherely.SpatialIndex(geoms)
    # replacing an element of the input array does not affect the index
    geoms[0] = spherely.create_point(40, 40)
    result = tree.query(spherely.create_point(0, 0), predicate="intersects")
    np.testing.assert_array_equal(result, [0])
    assert spherely.equals(tree.geometries[0], spherely.create_point(0, 0))


def test_is_built(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert tree.is_built is False
    tree.build()
    assert tree.is_built is True


def test_is_built_after_query(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert tree.is_built is False
    # the first query builds the index
    tree.query(spherely.create_point(1, 1))
    assert tree.is_built is True


def test_build_is_idempotent(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    assert not tree.is_built
    tree.build()
    assert tree.is_built
    tree.build()
    # a budget override on an already-built index is accepted and ignored
    tree.build(tmp_memory_budget=1)
    assert tree.is_built

    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])
    np.testing.assert_array_equal(tree.query(poly), [0, 1, 2])


def test_build_after_query(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    expected = tree.query(point, predicate="intersects")
    # the query already built the index; building again changes nothing
    assert tree.is_built
    tree.build()
    assert tree.is_built
    np.testing.assert_array_equal(tree.query(point, predicate="intersects"), expected)


def test_build_does_not_change_results(geographies: npt.NDArray[Any]) -> None:
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    lazy = spherely.SpatialIndex(geographies)
    eager = spherely.SpatialIndex(geographies)
    eager.build()
    assert eager.is_built
    assert not lazy.is_built

    np.testing.assert_array_equal(eager.query(poly), lazy.query(poly))
    np.testing.assert_array_equal(
        eager.query(poly, predicate="contains"), lazy.query(poly, predicate="contains")
    )


# the s2 default and two raised budgets -- raising is the only useful
# direction, see the SpatialIndex.build docstring
@pytest.mark.parametrize("budget", [104_857_600, 1024**3, 4 * 1024**3])
def test_build_budget_does_not_change_matches(
    geographies: npt.NDArray[Any], budget: int
) -> None:
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    default = spherely.SpatialIndex(geographies)
    default.build()
    tuned = spherely.SpatialIndex(geographies)
    tuned.build(tmp_memory_budget=budget)
    assert tuned.is_built

    # the budget moves s2's batch boundaries and with them the subdivision of
    # the index, so the unrefined candidate set is only guaranteed to stay a
    # superset of the matches -- what is invariant is the refined answer
    matches = tuned.query(poly, predicate="contains")
    np.testing.assert_array_equal(matches, default.query(poly, predicate="contains"))
    assert set(matches) <= set(tuned.query(poly))


def test_build_budget_below_default_is_legal(geographies: npt.NDArray[Any]) -> None:
    # a budget far below the default is accepted and still correct; it only
    # makes the build slower (more batches), so it is not a value to copy
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    default = spherely.SpatialIndex(geographies)
    default.build()
    tiny = spherely.SpatialIndex(geographies)
    tiny.build(tmp_memory_budget=1)
    assert tiny.is_built

    matches = tiny.query(poly, predicate="contains")
    np.testing.assert_array_equal(matches, default.query(poly, predicate="contains"))
    assert set(matches) <= set(tiny.query(poly))


def _overlapping_geographies() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # 600 overlapping 30-gons and 50 polygon queries over the same patch: big
    # enough that the coarse candidate set holds real false positives, and that
    # a budget below the default splits the build into several batches (18000
    # edges at s2geometry's 226 temporary bytes per edge is about 4 MB)
    rng = np.random.default_rng(7)

    def polygon(
        cx: float, cy: float, radius: float, nvertices: int
    ) -> spherely.Geography:
        angles = np.linspace(0, 2 * np.pi, nvertices, endpoint=False)
        coords = [
            (cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles.tolist()
        ]
        return spherely.create_polygon(coords + [coords[0]])

    indexed = np.array(
        [polygon(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.3, 30) for _ in range(600)]
    )
    queries = np.array(
        [polygon(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.2, 8) for _ in range(50)]
    )
    return indexed, queries


def test_build_budget_does_not_change_matches_on_larger_collection() -> None:
    indexed, queries = _overlapping_geographies()

    default = spherely.SpatialIndex(indexed)
    default.build()
    expected = default.query(queries, predicate="intersects")

    # unlike the four-geography fixture above, here the coarse candidate set is
    # strictly larger than the matches, so "superset of the matches" is a real
    # constraint rather than an accident of the collection's size
    assert _pairs(expected) < _pairs(default.query(queries))

    # 1 MiB forces the build into several batches (this collection needs about
    # 4 MB of temporary space) while 4 GiB builds it in one, as the default
    # does here: the candidate sets need not agree, the matches do
    for budget in (1024**2, 4 * 1024**3):
        tuned = spherely.SpatialIndex(indexed)
        tuned.build(tmp_memory_budget=budget)
        matches = tuned.query(queries, predicate="intersects")
        np.testing.assert_array_equal(matches, expected)
        assert _pairs(matches) <= _pairs(tuned.query(queries))


@pytest.mark.parametrize("budget", [0, -1, -(1024**3)])
def test_build_invalid_budget(geographies: npt.NDArray[Any], budget: int) -> None:
    tree = spherely.SpatialIndex(geographies)
    with pytest.raises(ValueError, match="positive"):
        tree.build(tmp_memory_budget=budget)
    assert not tree.is_built

    # the failed call left the index buildable
    tree.build()
    assert tree.is_built
    np.testing.assert_array_equal(
        tree.query(spherely.create_point(1, 1), predicate="intersects"), [1]
    )

    # the budget is validated before the "already built" early return, so an
    # invalid value is rejected even when there is nothing left to build
    with pytest.raises(ValueError, match="positive"):
        tree.build(tmp_memory_budget=budget)


def test_build_budget_out_of_range(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    # the budget is a C int64; anything larger is rejected by the binding
    with pytest.raises(TypeError):
        tree.build(tmp_memory_budget=2**63)
    assert not tree.is_built


def test_build_budget_is_restored(geographies: npt.NDArray[Any]) -> None:
    # the override is scoped to a single build: the process-wide s2 setting is
    # back to its previous value once the call returns
    before = spherely._s2_tmp_memory_budget()

    tuned = spherely.SpatialIndex(geographies)
    # deliberately different from the budget in effect, so that a failure to
    # put the previous value back is visible here
    tuned.build(tmp_memory_budget=before + 4096)
    assert spherely._s2_tmp_memory_budget() == before

    # ...so an index built afterwards without one behaves exactly as if the
    # override had never happened
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])
    plain = spherely.SpatialIndex(geographies)
    plain.build()
    assert spherely._s2_tmp_memory_budget() == before
    np.testing.assert_array_equal(
        plain.query(poly, predicate="contains"),
        tuned.query(poly, predicate="contains"),
    )


@pytest.fixture
def restore_s2_budget() -> Iterator[None]:
    # for the tests below, which move the process-wide s2 setting themselves
    before = spherely._s2_tmp_memory_budget()
    yield
    spherely._set_s2_tmp_memory_budget(before)


def test_s2_tmp_memory_budget_round_trip(restore_s2_budget: None) -> None:
    # the accessors reach the real s2 flag: what is set is what is read back
    spherely._set_s2_tmp_memory_budget(4096)
    assert spherely._s2_tmp_memory_budget() == 4096
    spherely._set_s2_tmp_memory_budget(8192)
    assert spherely._s2_tmp_memory_budget() == 8192


def test_build_budget_restores_previous_value(
    geographies: npt.NDArray[Any], restore_s2_budget: None
) -> None:
    # test_build_budget_is_restored runs at s2geometry's default, where putting
    # the previous value back and resetting to the default are the same thing;
    # start from a non-default value so that only the former passes
    spherely._set_s2_tmp_memory_budget(4096)

    tree = spherely.SpatialIndex(geographies)
    tree.build(tmp_memory_budget=8192)
    assert tree.is_built
    assert spherely._s2_tmp_memory_budget() == 4096


def test_build_empty_index() -> None:
    tree = spherely.SpatialIndex([])
    # nothing was queued, so there is nothing to build
    assert tree.is_built
    tree.build(tmp_memory_budget=1024)
    assert len(tree) == 0
    np.testing.assert_array_equal(tree.query(spherely.create_point(0, 0)), [])


def test_query_scalar(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    poly = spherely.create_polygon([(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)])

    result = tree.query(poly)
    assert result.dtype == np.intp
    # the three nearby points are candidates; the far point is not
    np.testing.assert_array_equal(result, [0, 1, 2])


def test_query_predicate_refines() -> None:
    # point 1 is outside the triangle (beyond the hypotenuse) but close enough
    # that it falls within the triangle's coarse cell covering; point 2 is far
    # away and is not a candidate at all
    geographies = np.array(
        [
            spherely.create_point(1, 1),
            spherely.create_point(4, 4),
            spherely.create_point(50, 50),
        ]
    )
    tree = spherely.SpatialIndex(geographies)
    triangle = spherely.create_polygon([(0, 0), (5, 0), (0, 5), (0, 0)])

    coarse = tree.query(triangle)
    refined = tree.query(triangle, predicate="contains")

    # the coarse candidate set includes the false positive, the refined
    # (exact) result does not
    np.testing.assert_array_equal(coarse, [0, 1])
    np.testing.assert_array_equal(refined, [0])


def test_query_predicate_intersects(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    result = tree.query(point, predicate="intersects")
    np.testing.assert_array_equal(result, [1])


def test_query_array(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    queries = np.array(
        [
            spherely.create_point(1, 1),
            spherely.create_point(50, 50),
        ]
    )
    result = tree.query(queries, predicate="intersects")
    # (input_index, tree_index) pairs
    np.testing.assert_array_equal(result, [[0, 1], [1, 3]])


def test_query_empty_geography_never_returned() -> None:
    geoms = np.array(
        [
            spherely.create_point(0, 0),
            spherely.create_polygon(None),  # empty
        ]
    )
    tree = spherely.SpatialIndex(geoms)
    assert len(tree) == 2
    poly = spherely.create_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)])
    result = tree.query(poly)
    np.testing.assert_array_equal(result, [0])


def test_query_disjoint_rejected(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    with pytest.raises(ValueError, match="disjoint"):
        tree.query(point, predicate="disjoint")  # type: ignore[call-overload]


def test_query_invalid_predicate(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    with pytest.raises(ValueError, match="invalid predicate"):
        tree.query(point, predicate="not_a_predicate")  # type: ignore[call-overload]


def test_query_nearest_scalar(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    result = tree.query_nearest(spherely.create_point(1.2, 1.2))
    assert result.dtype == np.intp
    np.testing.assert_array_equal(result, [1])


def test_query_nearest_ties() -> None:
    # two points exactly symmetric about the query point on the equator
    tree = spherely.SpatialIndex(
        [
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
            spherely.create_point(50, 0),
        ]
    )
    query = spherely.create_point(0, 0)
    np.testing.assert_array_equal(tree.query_nearest(query), [0, 1])

    single = tree.query_nearest(query, all_matches=False)
    np.testing.assert_array_equal(single, [0])


def test_query_nearest_ties_lowest_index() -> None:
    # more than two geographies tied at the minimum distance: all_matches=False
    # returns the lowest index, not an arbitrary member of the tie
    tree = spherely.SpatialIndex(
        [
            spherely.create_point(50, 0),  # far away
            spherely.create_point(2, 0),
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),  # duplicate of index 1
        ]
    )
    query = spherely.create_point(0, 0)

    np.testing.assert_array_equal(tree.query_nearest(query), [1, 2, 3])
    np.testing.assert_array_equal(tree.query_nearest(query, all_matches=False), [1])

    # array form (and with the distances)
    queries = np.array([query, spherely.create_point(49, 0)])
    single, distances = tree.query_nearest(
        queries, all_matches=False, return_distance=True
    )
    np.testing.assert_array_equal(single, [[0, 1], [1, 0]])
    np.testing.assert_allclose(
        distances,
        [
            spherely.distance(query, spherely.create_point(2, 0)),
            spherely.distance(queries[1], spherely.create_point(50, 0)),
        ],
    )


def test_query_nearest_exclusive_ties_lowest_index() -> None:
    # same, on the exclusive path: the geographies equal to the query are
    # dropped and the lowest of the remaining tie is returned
    tree = spherely.SpatialIndex(
        [
            spherely.create_point(0, 0),  # equal to the query
            spherely.create_point(2, 0),
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
        ]
    )
    query = spherely.create_point(0, 0)
    np.testing.assert_array_equal(tree.query_nearest(query, exclusive=True), [1, 2, 3])
    np.testing.assert_array_equal(
        tree.query_nearest(query, exclusive=True, all_matches=False), [1]
    )

    # ... and on the "everything at distance zero is equal" fallback path
    tree = spherely.SpatialIndex(
        [
            spherely.create_point(0, 0),
            spherely.create_point(0, 0),
            spherely.create_point(2, 0),
            spherely.create_point(-2, 0),
        ]
    )
    np.testing.assert_array_equal(tree.query_nearest(query, exclusive=True), [2, 3])
    np.testing.assert_array_equal(
        tree.query_nearest(query, exclusive=True, all_matches=False), [2]
    )


def test_query_nearest_interior_distance_zero() -> None:
    # a point inside an indexed polygon is at distance zero from it,
    # consistent with spherely.distance
    poly = spherely.create_polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    tree = spherely.SpatialIndex([poly, spherely.create_point(5.1, 5.1)])
    inside = spherely.create_point(5, 5)
    assert spherely.distance(inside, poly) == 0
    np.testing.assert_array_equal(tree.query_nearest(inside), [0])

    # and symmetrically: an indexed point inside a query polygon
    tree = spherely.SpatialIndex(
        [spherely.create_point(5, 5), spherely.create_point(20, 20)]
    )
    query_poly = spherely.create_polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
    np.testing.assert_array_equal(tree.query_nearest(query_poly), [0])


def test_query_nearest_empty_geographies() -> None:
    tree = spherely.SpatialIndex(
        [
            spherely.create_polygon(None),  # empty, never returned
            spherely.create_point(10, 10),
        ]
    )
    np.testing.assert_array_equal(tree.query_nearest(spherely.create_point(0, 0)), [1])

    # empty query geography -> no result
    result = tree.query_nearest(spherely.create_polygon(None))
    assert result.shape == (0,)

    # empty tree -> no result
    tree = spherely.SpatialIndex(np.array([], dtype=object))
    result = tree.query_nearest(spherely.create_point(0, 0))
    assert result.shape == (0,)


def test_query_nearest_return_distance(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    query = spherely.create_point(1.2, 1.2)

    indices, distances = tree.query_nearest(query, return_distance=True)
    np.testing.assert_array_equal(indices, [1])
    assert distances.dtype == np.float64
    np.testing.assert_allclose(distances, spherely.distance(query, geographies[1]))

    # the arguments may also be passed positionally, as in shapely
    positional = tree.query_nearest(query, None, True)
    np.testing.assert_array_equal(positional[0], indices)
    np.testing.assert_array_equal(positional[1], distances)

    # a return_distance only known at runtime is typed as the union of both
    flag = bool(len(indices))
    runtime = tree.query_nearest(query, return_distance=flag)
    assert isinstance(runtime, tuple)

    # radius scales the returned distances
    _, unscaled = tree.query_nearest(query, return_distance=True, radius=1)
    np.testing.assert_allclose(
        unscaled * spherely.EARTH_RADIUS_METERS, distances, rtol=1e-15
    )


def test_query_nearest_array(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    queries = np.array(
        [
            spherely.create_point(1.2, 1.2),
            spherely.create_point(49, 49),
        ]
    )

    result = tree.query_nearest(queries)
    assert result.dtype == np.intp
    # (input_index, tree_index) pairs
    np.testing.assert_array_equal(result, [[0, 1], [1, 3]])

    pairs, distances = tree.query_nearest(queries, return_distance=True)
    np.testing.assert_array_equal(pairs, result)
    expected = [
        spherely.distance(queries[0], geographies[1]),
        spherely.distance(queries[1], geographies[3]),
    ]
    np.testing.assert_allclose(distances, expected)


def test_query_nearest_array_ties_and_empty() -> None:
    tree = spherely.SpatialIndex(
        [
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
            spherely.create_point(50, 0),
        ]
    )
    queries = np.array(
        [
            spherely.create_point(0, 0),  # tied between 0 and 1
            spherely.create_polygon(None),  # empty: contributes no pairs
            spherely.create_point(49, 0),
        ]
    )
    result = tree.query_nearest(queries)
    np.testing.assert_array_equal(result, [[0, 0, 2], [0, 1, 2]])

    single = tree.query_nearest(queries, all_matches=False)
    # ties are broken deterministically on the lowest tree index
    np.testing.assert_array_equal(single, [[0, 2], [0, 2]])


def test_query_nearest_max_distance(geographies: npt.NDArray[Any]) -> None:
    tree = spherely.SpatialIndex(geographies)
    query = spherely.create_point(1.2, 1.2)
    dist = spherely.distance(query, geographies[1])

    np.testing.assert_array_equal(
        tree.query_nearest(query, max_distance=dist * 1.01), [1]
    )
    # inclusive bound
    np.testing.assert_array_equal(tree.query_nearest(query, max_distance=dist), [1])

    # no geography within max_distance -> empty result (shapely semantics)
    result = tree.query_nearest(query, max_distance=1.0)
    assert result.shape == (0,)
    indices, distances = tree.query_nearest(
        query, max_distance=1.0, return_distance=True
    )
    assert indices.shape == (0,)
    assert distances.shape == (0,)

    # array form
    queries = np.array([query, spherely.create_point(49, 49)])
    result = tree.query_nearest(queries, max_distance=dist * 1.01)
    np.testing.assert_array_equal(result, [[0], [1]])

    # array form with no match at all: (2, 0) pairs and 0 distances
    pairs, distances = tree.query_nearest(
        queries, max_distance=1.0, return_distance=True
    )
    assert pairs.shape == (2, 0)
    assert pairs.dtype == np.intp
    assert distances.shape == (0,)
    assert distances.dtype == np.float64


def test_query_nearest_max_distance_inclusive_boundary() -> None:
    # the distance returned by query_nearest must itself be accepted as an
    # (inclusive) max_distance: a fixture or two would not catch the ULP-level
    # round trip error between meters and chord angles, so sweep many pairs
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 50)
    lat = rng.uniform(-90, 90, 50)
    tree = spherely.SpatialIndex(np.asarray(spherely.points(lon, lat)))

    qlon = rng.uniform(-180, 180, 200)
    qlat = rng.uniform(-90, 90, 200)
    for query in np.asarray(spherely.points(qlon, qlat)):
        expected, distances = tree.query_nearest(query, return_distance=True)
        result = tree.query_nearest(query, max_distance=distances[0])
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("max_distance", [0, -1.0, np.nan, np.inf, -np.inf])
def test_query_nearest_invalid_max_distance(
    geographies: npt.NDArray[Any], max_distance: float
) -> None:
    # NaN in particular used to be silently accepted as "no bound"
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    match = "max_distance must be a finite value greater than 0"
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, max_distance=max_distance)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(np.array([point]), max_distance=max_distance)


@pytest.mark.parametrize("radius", [0, -1.0, np.nan, np.inf])
def test_query_nearest_invalid_radius(
    geographies: npt.NDArray[Any], radius: float
) -> None:
    # the radius filters here (it scales max_distance): a zero radius would
    # make every distance zero, a negative one would flip the chord angle
    tree = spherely.SpatialIndex(geographies)
    point = spherely.create_point(1, 1)
    match = "radius must be a finite value greater than 0"
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, radius=radius)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(point, max_distance=1000.0, radius=radius)
    with pytest.raises(ValueError, match=match):
        tree.query_nearest(np.array([point]), radius=radius)


def test_query_nearest_exclusive() -> None:
    tree = spherely.SpatialIndex(
        [
            spherely.create_point(0, 0),
            spherely.create_point(-2, 0),
            spherely.create_point(2, 0),
        ]
    )
    query = spherely.create_point(0, 0)
    np.testing.assert_array_equal(tree.query_nearest(query), [0])
    result = tree.query_nearest(query, exclusive=True)
    # the equal geography is skipped, the two symmetric neighbors are tied
    np.testing.assert_array_equal(result, [1, 2])

    # a non-equal geography at distance zero is still returned
    query = spherely.create_point(1, 1)
    poly = spherely.create_polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    tree = spherely.SpatialIndex([spherely.create_point(1, 1), poly])
    result, distances = tree.query_nearest(query, exclusive=True, return_distance=True)
    np.testing.assert_array_equal(result, [1])
    np.testing.assert_array_equal(distances, [0.0])


def test_query_nearest_exclusive_all_equal() -> None:
    # all geographies at distance zero are equal to the query
    tree = spherely.SpatialIndex(
        [spherely.create_point(1, 1), spherely.create_point(1, 1)]
    )
    query = spherely.create_point(1, 1)
    result = tree.query_nearest(query, exclusive=True)
    assert result.shape == (0,)

    # with a farther non-equal geography, that one is returned
    far = spherely.create_point(3, 3)
    tree = spherely.SpatialIndex(
        [spherely.create_point(1, 1), spherely.create_point(1, 1), far]
    )
    indices, distances = tree.query_nearest(query, exclusive=True, return_distance=True)
    np.testing.assert_array_equal(indices, [2])
    np.testing.assert_allclose(distances, spherely.distance(query, far))

    # unless it is beyond max_distance
    result = tree.query_nearest(query, exclusive=True, max_distance=1.0)
    assert result.shape == (0,)


def test_query_nearest_exclusive_all_equal_beyond_batch() -> None:
    # more geographies equal to the query than the initial max_results batch
    # of the fallback path (16), so the query has to be re-run with a larger
    # batch until the first result at a distance > 0 shows up
    query = spherely.create_point(1, 1)
    equal = [spherely.create_point(1, 1) for _ in range(40)]
    far = spherely.create_point(3, 3)
    tree = spherely.SpatialIndex(equal + [far])

    indices, distances = tree.query_nearest(query, exclusive=True, return_distance=True)
    np.testing.assert_array_equal(indices, [len(equal)])
    np.testing.assert_allclose(distances, spherely.distance(query, far))

    # ... and with nothing else in the index, the loop still terminates
    tree = spherely.SpatialIndex(equal)
    assert tree.query_nearest(query, exclusive=True).shape == (0,)


def test_query_nearest_against_distance_oracle() -> None:
    rng = np.random.default_rng(42)
    lon = rng.uniform(-180, 180, 20)
    lat = rng.uniform(-90, 90, 20)
    geographies = np.asarray(spherely.points(lon, lat))
    tree = spherely.SpatialIndex(geographies)

    for qlon, qlat in [(0, 0), (179.5, 0.5), (-179.5, -0.5), (0, 89.9), (0, -89.9)]:
        query = spherely.create_point(qlon, qlat)
        result = tree.query_nearest(query)
        distances = np.asarray(spherely.distance(query, geographies))
        expected = np.flatnonzero(distances == distances.min())
        np.testing.assert_array_equal(result, expected)
