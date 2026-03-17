import json
import pytest

import geopandas as gpd
import shapely.geometry as sg

from .context import gtfs_kit, DATA_DIR, cairns, cairns_shapeless
from gtfs_kit import shapes as gks
from gtfs_kit import constants as cs


def test_append_dist_to_shapes():
    feed1 = cairns.copy()
    s1 = feed1.shapes
    feed2 = gks.append_dist_to_shapes(feed1)
    s2 = feed2.shapes
    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = list(s1.columns.values) + ["shape_dist_traveled"]
    cols2 = list(s2.columns.values)
    assert set(cols1) == set(cols2)

    # Check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for name, group in s2.groupby("shape_id"):
        sdt = list(group["shape_dist_traveled"].values)
        assert sdt == sorted(sdt)


def test_geometrize_shapes():
    shapes = cairns.shapes.copy()
    geo_shapes = gks.geometrize_shapes(shapes)
    # Should be a GeoDataFrame
    assert isinstance(geo_shapes, gpd.GeoDataFrame)
    assert geo_shapes.crs == cs.WGS84
    # Should have the correct shape
    assert geo_shapes.shape[0] == shapes["shape_id"].nunique()
    assert geo_shapes.shape[1] == shapes.shape[1] - 2
    # Should have the correct columns
    expect_cols = set(list(shapes.columns) + ["geometry"]) - set(
        [
            "shape_pt_lon",
            "shape_pt_lat",
            "shape_pt_sequence",
            "shape_dist_traveled",
        ]
    )
    assert set(geo_shapes.columns) == expect_cols
    # A shape with only one point
    shapes = cairns.shapes.iloc[:1]
    geo_shapes = gks.geometrize_shapes(shapes)
    assert isinstance(geo_shapes, gpd.GeoDataFrame)
    assert geo_shapes.crs == cs.WGS84


def test_ungeometrize_shapes():
    shapes = cairns.shapes.copy()
    geo_shapes = gks.geometrize_shapes(shapes)
    print(geo_shapes)
    shapes2 = gks.ungeometrize_shapes(geo_shapes)

    # Test columns are correct
    expect_cols = set(list(shapes.columns)) - set(["shape_dist_traveled"])
    assert set(shapes2.columns) == expect_cols

    # Data frames should agree on certain columns
    cols = ["shape_id", "shape_pt_lon", "shape_pt_lat"]
    print(shapes[cols])
    assert shapes2[cols].equals(shapes[cols])


def test_get_shapes():
    g = gks.get_shapes(cairns, as_gdf=True)
    assert g.crs == cs.WGS84
    assert set(g.columns) == {"shape_id", "geometry"}
    assert gks.get_shapes(cairns_shapeless, as_gdf=True) is None


def test_build_geometry_by_shape():
    d = gks.build_geometry_by_shape(cairns)
    assert isinstance(d, dict)
    assert len(d) == cairns.shapes.shape_id.nunique()
    assert gks.build_geometry_by_shape(cairns_shapeless) == {}


def test_shapes_to_geojson():
    feed = cairns.copy()
    shape_ids = feed.shapes.shape_id.unique()[:2]
    collection = gks.shapes_to_geojson(feed, shape_ids)
    assert isinstance(collection, dict)
    assert len(collection["features"]) == len(shape_ids)

    assert gks.shapes_to_geojson(cairns_shapeless) == {
        "type": "FeatureCollection",
        "features": [],
    }


def test_get_shapes_intersecting_geometry():
    feed = cairns.copy()
    path = DATA_DIR / "cairns_square_stop_750070.geojson"
    polygon = sg.shape(json.load(path.open())["features"][0]["geometry"])
    pshapes = gks.get_shapes_intersecting_geometry(feed, polygon)
    shape_ids = ["120N0005", "1200010", "1200001"]
    assert set(pshapes["shape_id"].unique()) == set(shape_ids)
    g = gks.get_shapes_intersecting_geometry(feed, polygon, as_gdf=True)
    assert g.crs == "epsg:4326"
    assert set(g["shape_id"].unique()) == set(shape_ids)
    assert gks.get_shapes_intersecting_geometry(cairns_shapeless, polygon) is None


def test_split_simple_0():
    def assert_simple_parts(name, parts, expected):
        assert isinstance(parts, list), f"{name}: must return list"
        assert len(parts) == len(expected), (
            f"{name}: expected {len(expected)} components, got {len(parts)}"
        )
        for i, (part, exp_coords) in enumerate(zip(parts, expected), start=1):
            assert isinstance(part, sg.LineString), f"{name} comp {i}: not a LineString"
            got = list(map(tuple, part.coords))
            assert got == exp_coords, (
                f"{name} comp {i}: coords mismatch.\nGot: {got}\nExp: {exp_coords}"
            )
            assert part.is_simple, f"{name} comp {i}: returned LineString not simple"

    # Test 1: straight line, no repeats -> single component
    line1 = sg.LineString([(0, 0), (1, 0), (2, 0)])
    expected1 = [[(0, 0), (1, 0), (2, 0)]]

    # Test 2: loop then tail (includes a consecutive duplicate that should be ignored)
    line2 = sg.LineString([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 1)])
    expected2 = [[(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0)], [(0, 0), (0, 1)]]

    # Test 3: doubles back on an interior vertex
    line3 = sg.LineString([(0, 0), (2, 0), (2, 1), (1, 1), (1, 0), (2, 0), (3, 0)])
    expected3 = [[(0, 0), (2, 0), (2, 1), (1, 1)], [(1, 1), (1, 0), (2, 0), (3, 0)]]

    assert_simple_parts("straight", gks.split_simple_0(line1), expected1)
    assert_simple_parts("loop_then_tail", gks.split_simple_0(line2), expected2)
    assert_simple_parts("double_back", gks.split_simple_0(line3), expected3)

    # Extra edge cases
    # 4) Figure-8 crossing
    line4 = sg.LineString([(0, 0), (2, 2), (0, 2), (2, 0)])
    exp4 = [[(0, 0), (2, 2), (0, 2)], [(0, 2), (2, 0)]]
    assert_simple_parts("figure8", gks.split_simple_0(line4), exp4)

    # 5) T-junction interior
    line5 = sg.LineString([(0, 0), (2, 0), (1, 0), (1, 1)])
    exp5 = [[(0, 0), (2, 0)], [(2, 0), (1, 0), (1, 1)]]
    assert_simple_parts("t_junction_interior", gks.split_simple_0(line5), exp5)

    # 6) Close loop then continue
    line6 = sg.LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0), (2, 0)])
    exp6 = [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], [(0, 0), (2, 0)]]
    assert_simple_parts("close_then_continue", gks.split_simple_0(line6), exp6)

    # 7) Overlapping collinear segments
    line7 = sg.LineString([(0, 0), (2, 0), (1, 0), (3, 0)])
    exp7 = [[(0, 0), (2, 0)], [(2, 0), (1, 0)], [(1, 0), (3, 0)]]
    assert_simple_parts("overlap_collinear", gks.split_simple_0(line7), exp7)

    # 8) Envelope-touch case (vertical through horizontal at boundary)
    line8 = sg.LineString([(0, 0), (2, 0), (1, 1), (1, -1)])
    exp8 = [[(0, 0), (2, 0), (1, 1)], [(1, 1), (1, -1)]]
    assert_simple_parts("envelope_touch", gks.split_simple_0(line8), exp8)

    # 9) Non-consecutive duplicate revisit
    line9 = sg.LineString([(0, 0), (1, 0), (1, 1), (1, 0), (2, 0)])
    exp9 = [[(0, 0), (1, 0), (1, 1)], [(1, 1), (1, 0), (2, 0)]]
    assert_simple_parts("nonconsecutive_dup_revisit", gks.split_simple_0(line9), exp9)


def test_split_simple():
    shapes_g = gks.get_shapes(cairns, as_gdf=True, use_utm=True).assign(
        length=lambda x: x.length,
        is_simple=lambda x: x.is_simple,
    )
    # We should have some non-simple shapes to start with
    assert not shapes_g["is_simple"].all()

    s = gks.split_simple(shapes_g)

    # Columns should be correct
    assert set(s.columns) == {
        "shape_id",
        "subshape_id",
        "subshape_sequence",
        "subshape_length_m",
        "cum_length_m",
        "geometry",
    }
    # All sublinestrings of result should be simple
    assert s.is_simple.all()

    # Check each shape group
    for shape_id, group in s.groupby("shape_id"):
        ss = shapes_g.loc[lambda x: x["shape_id"] == shape_id]
        # Each subshape should be shorter than shape
        assert (group["subshape_length_m"] <= ss["length"].sum()).all()
        # Cumulative length should equal shape length within 0.1%
        L = ss["length"].iat[0]
        assert group["cum_length_m"].max() == pytest.approx(L, rel=0.001)

    # Create a (non-simple) bow-tie
    bowtie = sg.LineString([(0, 0), (1, 1), (0, 1), (1, 0)])
    g = gpd.GeoDataFrame(
        {"shape_id": ["test_shape"], "geometry": [bowtie]}, crs="EPSG:2193"
    )

    result = gks.split_simple(g)
    # No sub-linestring should have only one coordinate
    for geom in result.geometry:
        assert len(geom.coords) > 1, f"Found a degenerate one-point LineString: {geom}"
