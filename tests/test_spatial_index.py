from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import spherely


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
