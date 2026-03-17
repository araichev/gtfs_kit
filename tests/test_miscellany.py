import geopandas as gp
import numpy as np
import pandas as pd
import pytest
import shapely.geometry as sg
from pandas.testing import assert_series_equal

from gtfs_kit import constants as gkc
from gtfs_kit import miscellany as gkm
from gtfs_kit import shapes as gks

from .context import (
    DATA_DIR,
    cairns,
    cairns_dates,
    cairns_trip_stats,
    gtfs_kit,
    nyc_subway,
    sample,
)


def test_list_fields():
    feed = sample.copy()

    with pytest.raises(ValueError):
        gkm.list_fields(feed, "bad_table")

    for table in [None, "stops"]:
        f = gkm.list_fields(feed, table)
        assert isinstance(f, pd.DataFrame)
        expect_cols = {
            "table",
            "column",
            "num_values",
            "num_nonnull_values",
            "num_unique_values",
            "min_value",
            "max_value",
        }
        assert set(f.columns) == expect_cols
        assert f.shape[0]


def test_describe():
    feed = sample.copy()  # No distances here
    a = gkm.describe(feed)
    assert isinstance(a, pd.DataFrame)
    assert set(a.columns) == set(["indicator", "value"])


def test_assess_quality():
    feed = sample.copy()  # No distances here
    a = gkm.assess_quality(feed)
    assert isinstance(a, pd.DataFrame)
    assert set(a.columns) == set(["indicator", "value"])


def test_convert_dist():
    # Test with no distances
    feed1 = cairns.copy()  # No distances here
    feed2 = gkm.convert_dist(feed1, "mi")
    assert feed2.dist_units == "mi"

    # Test with distances and identity conversion
    feed1 = gks.append_dist_to_shapes(feed1)
    feed2 = gkm.convert_dist(feed1, feed1.dist_units)
    assert feed1 == feed2

    # Test with proper conversion
    feed2 = gkm.convert_dist(feed1, "m")
    assert_series_equal(
        feed2.shapes["shape_dist_traveled"] / 1000, feed1.shapes["shape_dist_traveled"]
    )


def test_compute_network_stats_0():
    feed = cairns.copy()
    trip_stats = cairns_trip_stats.copy()
    trip_stats.loc[0, "route_type"] = 2  # Add another route type besides 3
    for split_route_types in [True, False]:
        f = gkm.compute_network_stats_0(
            feed.stop_times, trip_stats, split_route_types=split_route_types
        )

        # Should have correct num rows
        n = 2 if split_route_types else 1
        assert f.shape[0] == n
        # Should contain the correct columns
        expect_cols = {
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "num_routes",
            "num_stops",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
        }
        if split_route_types:
            expect_cols.add("route_type")

        assert set(f.columns) == expect_cols


def test_compute_network_stats():
    feed = cairns.copy()
    dates = cairns_dates
    trip_stats = cairns_trip_stats.copy()
    n = trip_stats.shape[0]
    trip_stats.loc[: n // 2, "route_type"] = 2  # Add another route type besides 3
    for split_route_types in [True, False]:
        f = gkm.compute_network_stats(
            feed, dates + ["19990101"], trip_stats, split_route_types=split_route_types
        )
        # Should have correct num rows
        n = 2 if split_route_types else 1
        assert f.shape[0] == n * len(dates)

        # Should have the correct dates
        assert set(f["date"].values) == set(cairns_dates)

        # Should have correct columns
        expect_cols = {
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "num_routes",
            "num_stops",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
            "date",
        }
        if split_route_types:
            expect_cols.add("route_type")

        assert set(f.columns) == expect_cols

        # Non-feed dates should yield empty DataFrame
        f = gkm.compute_network_stats(
            feed, ["19990101"], trip_stats, split_route_types=split_route_types
        )
        assert f.empty


def test_compute_network_time_series():
    feed = cairns.copy()
    dates = cairns_dates
    n = feed.routes.shape[0]
    feed.routes.loc[: n // 2, "route_type"] = 2  # Add another route type besides 3

    for split_route_types in [True, False]:
        f = gkm.compute_network_time_series(
            feed,
            dates + ["19990101"],
            freq="12h",
            split_route_types=split_route_types,
        )

        # Should have correct num rows
        n = 2 if split_route_types else 1
        assert len(f) == n * len(dates) * 2

        # Should have correct columns
        expect_cols = {
            "datetime",
            "num_trip_starts",
            "num_trip_ends",
            "num_trips",
            "service_distance",
            "service_duration",
            "service_speed",
        }
        if split_route_types:
            expect_cols.add("route_type")

        assert set(f.columns) == expect_cols

        # Should have correct dates
        assert set(f["datetime"].dt.strftime("%Y%m%d")) == set(dates)

        # Empty check
        f = gkm.compute_network_time_series(
            feed, ["19990101"], split_route_types=split_route_types
        )
        assert f.empty


def test_create_shapes():
    feed1 = cairns.copy()
    # Remove a trip shape
    trip_id = "CNS2014-CNS_MUL-Weekday-00-4165878"
    feed1.trips.loc[feed1.trips["trip_id"] == trip_id, "shape_id"] = np.nan
    feed2 = gkm.create_shapes(feed1)
    # Should create only 1 new shape
    assert len(set(feed2.shapes["shape_id"]) - set(feed1.shapes["shape_id"])) == 1

    feed2 = gkm.create_shapes(feed1, all_trips=True)
    # Number of shapes should equal number of unique stop sequences
    st = feed1.stop_times.sort_values(["trip_id", "stop_sequence"])
    stop_seqs = set(
        [tuple(group["stop_id"].values) for __, group in st.groupby("trip_id")]
    )
    assert feed2.shapes["shape_id"].nunique() == len(stop_seqs)


def test_compute_bounds():
    feed = cairns.copy()
    minlon, minlat, maxlon, maxlat = gkm.compute_bounds(feed)
    # Bounds should be in the ball park
    assert 145 < minlon < 146
    assert 145 < maxlon < 146
    assert -18 < minlat < -15
    assert -18 < maxlat < -15

    # A one-stop bounds should be the stop
    stop_id = feed.stops.stop_id.iat[0]
    minlon, minlat, maxlon, maxlat = gkm.compute_bounds(feed, [stop_id])
    expect_lon, expect_lat = feed.stops.loc[
        lambda x: x.stop_id == stop_id, ["stop_lon", "stop_lat"]
    ].values[0]
    assert minlon == expect_lon
    assert minlat == expect_lat
    assert maxlon == expect_lon
    assert maxlat == expect_lat


def test_compute_convex_hull():
    feed = cairns.copy()
    hull = gkm.compute_convex_hull(feed)
    assert isinstance(hull, sg.Polygon)
    # Hull should encompass all stops
    m = sg.MultiPoint(feed.stops[["stop_lon", "stop_lat"]].values)
    assert hull.contains(m)

    # A one-stop hull should be the stop
    stop_id = feed.stops.stop_id.iat[0]
    hull = gkm.compute_convex_hull(feed, [stop_id])
    lon, lat = np.array(hull.coords)[0]
    expect_lon, expect_lat = feed.stops.loc[
        lambda x: x.stop_id == stop_id, ["stop_lon", "stop_lat"]
    ].values[0]
    assert lon == expect_lon
    assert lat == expect_lat


def test_compute_centroid():
    feed = cairns.copy()
    centroid = gkm.compute_centroid(feed)
    assert isinstance(centroid, sg.Point)
    # Centroid should lie within bounds
    lon, lat = centroid.coords[0]
    bounds = gkm.compute_bounds(feed)
    assert bounds[0] < lon < bounds[2]
    assert bounds[1] < lat < bounds[3]

    # A one-stop centroid should be the stop
    stop_id = feed.stops.stop_id.iat[0]
    centroid = gkm.compute_centroid(feed, [stop_id])
    lon, lat = centroid.coords[0]
    expect_lon, expect_lat = feed.stops.loc[
        lambda x: x.stop_id == stop_id, ["stop_lon", "stop_lat"]
    ].values[0]
    assert lon == expect_lon
    assert lat == expect_lat


def test_restrict_to_dates():
    feed1 = cairns.copy()
    dates = feed1.get_first_week()[6:]
    feed2 = gkm.restrict_to_dates(feed1, dates)
    # Should have correct agency
    assert feed2.agency.equals(feed1.agency)
    # Should have correct dates
    assert set(feed2.get_dates()) < set(feed1.get_dates())
    # Should have correct trips
    assert set(feed2.trips.trip_id) < set(feed1.trips.trip_id)
    # Should have correct routes
    assert set(feed2.routes.route_id) < set(feed1.routes.route_id)
    # Should have correct shapes
    assert set(feed2.shapes.shape_id) < set(feed1.shapes.shape_id)
    # Should have correct stops
    assert set(feed2.stops.stop_id) < set(feed1.stops.stop_id)

    # Try again with date out of range
    dates = ["20180101"]
    feed2 = gkm.restrict_to_dates(feed1, dates)
    assert feed2.agency.equals(feed1.agency)
    assert feed2.trips.empty
    assert feed2.routes.empty
    assert feed2.shapes.empty
    assert feed2.stops.empty
    assert feed2.stop_times.empty


def test_restrict_to_trips():
    feed1 = cairns
    trip_ids = feed1.trips["trip_id"][:2].values
    feed2 = gkm.restrict_to_trips(feed1, trip_ids)
    # Should have correct trips
    assert set(feed2.trips["trip_id"]) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips[feed1.trips["trip_id"].isin(trip_ids)]["shape_id"]
    assert set(feed2.shapes["shape_id"]) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times[feed1.stop_times["trip_id"].isin(trip_ids)]["stop_id"]
    assert set(feed2.stop_times["stop_id"]) == set(stop_ids)

    feed2 = gkm.restrict_to_trips(feed1, ["fake"])
    # All non-agency tables should be empty
    for table in gkc.DTYPES:
        if table != "agency" and getattr(feed1, table) is not None:
            assert getattr(feed2, table).empty

    feed = nyc_subway
    # Grab a stop with a parent station
    stop_id = feed.stops.loc[lambda x: x["parent_station"].notna(), "stop_id"].iat[0]
    parent_id = feed.stops.loc[lambda x: x["stop_id"] == stop_id, "parent_station"].iat[0]
    # Get one of its trips
    trip_id = feed.stop_times.loc[lambda x: x["stop_id"] == stop_id, "trip_id"].iat[0]
    # Restrict feed to that trip and resulting feed should contain that parent station
    feed2 = feed.restrict_to_trips([trip_id])
    assert parent_id in feed2.stops["stop_id"].values
    assert parent_id in feed2.transfers["from_stop_id"].values

    # Should not throw error when parent station absent
    feed1 = cairns.copy()
    del feed1.stops["parent_station"]
    trip_ids = feed1.trips["trip_id"][:2].values
    feed2 = gkm.restrict_to_trips(feed1, trip_ids)
    assert set(feed2.trips["trip_id"]) == set(trip_ids)


def test_restrict_to_routes():
    feed1 = cairns.copy()
    route_ids = feed1.routes["route_id"][:2].tolist()
    feed2 = gkm.restrict_to_routes(feed1, route_ids)
    # Should have correct routes
    assert set(feed2.routes["route_id"]) == set(route_ids)
    # Should have correct trips
    trip_ids = feed1.trips[feed1.trips["route_id"].isin(route_ids)]["trip_id"]
    assert set(feed2.trips["trip_id"]) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips[feed1.trips["trip_id"].isin(trip_ids)]["shape_id"]
    assert set(feed2.shapes["shape_id"]) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times[feed1.stop_times["trip_id"].isin(trip_ids)]["stop_id"]
    assert set(feed2.stop_times["stop_id"]) == set(stop_ids)


def test_restrict_to_agencies():
    feed1 = cairns.copy()
    # Most tests should be covered by `test_restrict_routes`
    # But will need to test for reduced agency and routes tables

    # Modify cairns to contain agency_id column
    new_routes = feed1.routes.copy()
    new_agency = feed1.agency.copy()

    # Modify cairns to include agency_id
    new_agency.insert(loc=0, column="agency_id", value="OP1")

    # Add new empty agency "OP2" for testing
    new_agency.loc[1] = (
        "OP2",
        "dummy agency",
        "agency_url",
        "GMT",
        "en",
        "(00)00000000",
    )

    # Modify cairns routes to contain new agency
    new_routes.insert(loc=0, column="agency_id", value="OP1")
    new_routes.loc[0, "agency_id"] = "OP2"

    # Reassign agency
    feed1.routes = new_routes
    feed1.agency = new_agency

    # Test function
    feed2 = gkm.restrict_to_agencies(feed1, ["OP2"])
    assert set(feed2.agency["agency_id"]) == {"OP2"}
    assert set(feed2.routes.iloc[0, 1:]) == set(feed1.routes.iloc[0, 1:])


def test_restrict_to_area():
    feed1 = cairns.copy()
    area = gp.read_file(DATA_DIR / "cairns_square_stop_750070.geojson")
    feed2 = gkm.restrict_to_area(feed1, area)
    # Should have correct routes
    rsns = ["120", "120N"]
    assert set(feed2.routes["route_short_name"]) == set(rsns)
    # Should have correct trips
    route_ids = feed1.routes[feed1.routes["route_short_name"].isin(rsns)]["route_id"]
    trip_ids = feed1.trips[feed1.trips["route_id"].isin(route_ids)]["trip_id"]
    assert set(feed2.trips["trip_id"]) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips[feed1.trips["trip_id"].isin(trip_ids)]["shape_id"]
    assert set(feed2.shapes["shape_id"]) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times[feed1.stop_times["trip_id"].isin(trip_ids)]["stop_id"]
    assert set(feed2.stop_times["stop_id"]) == set(stop_ids)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_screen_line_counts():
    feed = cairns.append_dist_to_stop_times()
    dates = cairns_dates[:2]

    # Load screen line
    path = DATA_DIR / "cairns_screen_lines.geojson"
    screen_lines = gp.read_file(path)
    f = gkm.compute_screen_line_counts(feed, screen_lines, dates)

    # Should have correct columns
    expect_cols = {
        "date",
        "screen_line_id",
        "trip_id",
        "direction_id",
        "route_id",
        "route_short_name",
        "route_type",
        "shape_id",
        "crossing_direction",
        "crossing_dist_m",
        "crossing_time",
    }
    assert set(f.columns) == expect_cols

    # Should have both directions
    assert set(f.crossing_direction.unique()) == {-1, 1}

    # Should only have feed dates
    assert set(f["date"].values) == set(dates)

    # Empty check
    f = gkm.compute_screen_line_counts(feed, screen_lines, ["20010101"])
    assert f.empty
