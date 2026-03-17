import itertools

import folium as fl
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from gtfs_kit import constants as cs
from gtfs_kit import routes as gkr

from .context import (
    DATA_DIR,
    cairns,
    cairns_dates,
    cairns_shapeless,
    cairns_trip_stats,
    gtfs_kit,
)

sample = gtfs_kit.read_feed(DATA_DIR / "sample_gtfs_2.zip", dist_units="km")


def test_get_routes():
    feed = cairns.copy()
    date = cairns_dates[0]
    f = gkr.get_routes(feed, date)
    # Should have the correct shape
    assert f.shape[0] <= feed.routes.shape[0]
    assert f.shape[1] == feed.routes.shape[1]
    # Should have correct columns
    assert set(f.columns) == set(feed.routes.columns)

    g = gkr.get_routes(feed, date, "07:30:00")
    # Should have the correct shape
    assert g.shape[0] <= f.shape[0]
    assert g.shape[1] == f.shape[1]
    # Should have correct columns
    assert set(g.columns) == set(feed.routes.columns)

    # Test geo options
    feed = cairns.copy()
    g = gkr.get_routes(feed, as_gdf=True, use_utm=True)
    assert isinstance(g, gpd.GeoDataFrame)
    assert g.crs != cs.WGS84

    g = gkr.get_routes(feed, as_gdf=True, split_directions=True)
    assert g.crs == cs.WGS84
    assert (
        g.shape[0] == feed.trips[["route_id", "direction_id"]].drop_duplicates().shape[0]
    )

    with pytest.raises(ValueError):
        gkr.get_routes(cairns_shapeless, as_gdf=True)

    # Test written by Gilles Cuyaubere
    feed = gtfs_kit.Feed(dist_units="km")
    feed.agency = pd.DataFrame(
        {"agency_id": ["agency_id_0"], "agency_name": ["agency_name_0"]}
    )
    feed.routes = pd.DataFrame(
        {
            "route_id": ["route_id_0"],
            "agency_id": ["agency_id_0"],
            "route_short_name": [None],
            "route_long_name": ["route_long_name_0"],
            "route_desc": [None],
            "route_type": [1],
            "route_url": [None],
            "route_color": [None],
            "route_text_color": [None],
        }
    )
    feed.trips = pd.DataFrame(
        {
            "route_id": ["route_id_0"],
            "service_id": ["service_id_0"],
            "trip_id": ["trip_id_0"],
            "trip_headsign": [None],
            "trip_short_name": [None],
            "direction_id": [None],
            "block_id": [None],
            "wheelchair_accessible": [None],
            "bikes_allowed": [None],
            "trip_desc": [None],
            "shape_id": ["shape_id_0"],
        }
    )
    feed.shapes = pd.DataFrame(
        {
            "shape_id": ["shape_id_0", "shape_id_0"],
            "shape_pt_lon": [2.36, 2.37],
            "shape_pt_lat": [48.82, 48.82],
            "shape_pt_sequence": [0, 1],
        }
    )
    feed.stops = pd.DataFrame(
        {
            "stop_id": ["stop_id_0", "stop_id_1"],
            "stop_name": ["stop_name_0", "stop_name_1"],
            "stop_desc": [None, None],
            "stop_lat": [48.82, 48.82],
            "stop_lon": [2.36, 2.37],
            "zone_id": [None, None],
            "stop_url": [None, None],
            "location_type": [0, 0],
            "parent_station": [None, None],
            "wheelchair_boarding": [None, None],
        }
    )
    feed.stop_times = pd.DataFrame(
        {
            "trip_id": ["trip_id_0", "trip_id_0"],
            "arrival_time": ["11:40:00", "11:45:00"],
            "departure_time": ["11:40:00", "11:45:00"],
            "stop_id": ["stop_id_0", "stop_id_1"],
            "stop_sequence": [0, 1],
            "stop_time_desc": [None, None],
            "pickup_type": [None, None],
            "drop_off_type": [None, None],
        }
    )

    g = gkr.get_routes(feed, as_gdf=True)
    assert g.crs == cs.WGS84

    # Turning a route's shapes into point geometries
    # should yield an empty route geometry and should not throw an error
    feed = cairns.copy()
    rid = feed.routes.loc[0, "route_id"]
    shids = feed.trips.loc[lambda x: x["route_id"] == rid, "shape_id"]
    f0 = feed.shapes.loc[lambda x: x["shape_id"].isin(shids)].drop_duplicates("shape_id")
    f1 = feed.shapes.loc[lambda x: ~x["shape_id"].isin(shids)]
    feed.shapes = pd.concat([f0, f1])
    assert (
        feed.get_routes(as_gdf=True)
        .loc[lambda x: x["route_id"] == rid, "geometry"]
        .iat[0]
        is None
    )

    # Turning a route's shapes into None geometries
    # should yield an empty route geometry and should not throw an error
    feed = cairns.copy()
    rid = feed.routes.loc[0, "route_id"]
    shids = feed.trips.loc[lambda x: x["route_id"] == rid, "shape_id"]
    feed.shapes = feed.shapes.loc[lambda x: ~x["shape_id"].isin(shids)]
    assert (
        feed.get_routes(as_gdf=True)
        .loc[lambda x: x["route_id"] == rid, "geometry"]
        .iat[0]
        is None
    )


def test_build_route_timetable():
    feed = sample.copy()
    route_id = feed.routes["route_id"].values[0]
    dates = feed.get_first_week()[:2]
    f = gkr.build_route_timetable(feed, route_id, dates)

    # Should have the correct columns
    expect_cols = set(feed.trips.columns) | set(feed.stop_times.columns) | set(["date"])
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert f.date.unique().tolist() == dates

    # Empty check
    f = gkr.build_route_timetable(feed, route_id, dates[2:])
    assert f.empty


def test_routes_to_geojson():
    feed = cairns.copy()
    n = 3
    route_ids = feed.routes["route_id"].to_list()[:n]
    route_short_names = feed.routes.loc[
        lambda x: x["route_id"] == route_ids[-1], "route_short_name"
    ]
    route_ids = route_ids[: n - 1]

    gj = gkr.routes_to_geojson(
        feed, route_ids=route_ids, route_short_names=route_short_names
    )
    assert len(gj["features"]) == n

    gj = gkr.routes_to_geojson(
        feed, route_ids=route_ids, route_short_names=route_short_names, include_stops=True
    )
    k = (
        feed.stop_times.merge(feed.trips.filter(["trip_id", "route_id"]))
        .loc[lambda x: x.route_id.isin(route_ids), "stop_id"]
        .nunique()
    )
    assert len(gj["features"]) == n + k

    with pytest.raises(ValueError):
        gkr.routes_to_geojson(cairns_shapeless)

    gj = gkr.routes_to_geojson(cairns, route_ids=["bingo"])
    assert not gj["features"]


def test_map_routes():
    feed = cairns.copy()
    rids = feed.routes["route_id"].iloc[:1]
    rsns = feed.routes["route_short_name"].iloc[-2:]
    m = gkr.map_routes(feed, route_ids=rids, route_short_names=rsns, show_stops=True)
    assert isinstance(m, fl.Map)

    with pytest.raises(ValueError):
        gkr.map_routes(feed)


@pytest.mark.slow
def test_compute_route_stats_0():
    feed = cairns.copy()
    ts1 = cairns_trip_stats.copy()
    ts2 = cairns_trip_stats.copy()
    ts2.direction_id = np.nan

    for ts, split_directions in itertools.product([ts1, ts2], [True, False]):
        if split_directions and ts.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                gkr.compute_route_stats_0(ts, split_directions=split_directions)
            continue

        rs = gkr.compute_route_stats_0(ts, split_directions=split_directions)

        # Should be a data frame of the correct shape
        if split_directions:
            max_num_routes = 2 * feed.routes.shape[0]
        else:
            max_num_routes = feed.routes.shape[0]
        assert rs.shape[0] <= max_num_routes

        # Should contain the correct columns
        expect_cols = set(
            [
                "route_id",
                "route_short_name",
                "route_type",
                "num_trips",
                "num_trip_ends",
                "num_trip_starts",
                "num_stop_patterns",
                "is_loop",
                "start_time",
                "end_time",
                "max_headway",
                "min_headway",
                "mean_headway",
                "peak_num_trips",
                "peak_start_time",
                "peak_end_time",
                "service_duration",
                "service_distance",
                "service_speed",
                "mean_trip_distance",
                "mean_trip_duration",
            ]
        )
        if split_directions:
            expect_cols.add("direction_id")
        else:
            expect_cols.add("is_bidirectional")
        assert set(rs.columns) == expect_cols

    # Empty check
    rs = gkr.compute_route_stats_0(pd.DataFrame(), split_directions=split_directions)
    assert rs.empty


def test_compute_route_stats():
    feed = cairns.copy()
    dates = cairns_dates + ["19990101"]
    n = 3
    rids = cairns_trip_stats.loc[:n, "route_id"]
    trip_stats = cairns_trip_stats.loc[lambda x: x["route_id"].isin(rids)]

    for split_directions in [True, False]:
        rs = gkr.compute_route_stats(
            feed, dates, trip_stats, split_directions=split_directions
        )

        # Should have correct num rows
        N = 2 * n * len(dates) if split_directions else n * len(dates)
        assert rs.shape[0] <= N

        # Should have correct columns
        expect_cols = {
            "date",
            "route_id",
            "route_short_name",
            "route_type",
            "num_trips",
            "num_trip_ends",
            "num_trip_starts",
            "num_stop_patterns",
            "is_loop",
            "start_time",
            "end_time",
            "max_headway",
            "min_headway",
            "mean_headway",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
            "mean_trip_distance",
            "mean_trip_duration",
        }
        if split_directions:
            expect_cols.add("direction_id")
        else:
            expect_cols.add("is_bidirectional")

        assert set(rs.columns) == expect_cols

        # Should only contains valid dates
        set(rs["date"].tolist()) == set(cairns_dates)

        # Empty dates should yield empty DataFrame
        rs = gkr.compute_route_stats(
            feed, ["19990101"], trip_stats, split_directions=split_directions
        )
        assert rs.empty


@pytest.mark.slow
def test_compute_route_time_series_0():
    ts1 = cairns_trip_stats.copy()
    ts2 = cairns_trip_stats.copy()
    ts2.direction_id = pd.NA
    nroutes = ts1["route_id"].nunique()
    for ts, split_directions in itertools.product([ts1, ts2], [True, False]):
        if split_directions and ts.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                gkr.compute_route_stats_0(ts, split_directions=split_directions)
            continue

        rs = gkr.compute_route_stats_0(ts, split_directions=split_directions)
        rts = gkr.compute_route_time_series_0(
            ts, split_directions=split_directions, freq="h"
        )

        # Should have correct num rows
        if split_directions:
            assert rts.shape[0] >= nroutes * 24
        else:
            assert rts.shape[0] == nroutes * 24

        # Should have correct columns
        expect_cols = {
            "datetime",
            "route_id",
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "service_distance",
            "service_duration",
            "service_speed",
        }
        if split_directions:
            expect_cols |= {"direction_id"}
        assert set(rts.columns) == expect_cols

        # Each route should have a correct service distance
        if not split_directions:
            for route_id, rsr in rs.groupby("route_id"):
                get = rts.loc[
                    lambda x: x["route_id"] == route_id, "service_distance"
                ].sum()
                expect = rsr["service_distance"].sum()
                assert get == pytest.approx(expect)

    # Empty check
    rts = gkr.compute_route_time_series_0(
        pd.DataFrame(), split_directions=split_directions, freq="1h"
    )
    assert rts.empty


def test_compute_route_time_series():
    feed = cairns.copy()
    dates = cairns_dates  # Spans three consecutive dates
    n = 3
    rids = feed.routes.loc[:n, "route_id"]
    trip_stats = cairns_trip_stats.loc[lambda x: x["route_id"].isin(rids)]

    for split_directions in [True, False]:
        rs = gkr.compute_route_stats(
            feed, dates, trip_stats, split_directions=split_directions
        )
        rts = gkr.compute_route_time_series(
            feed,
            dates + ["19990101"],  # Invalid last date
            trip_stats,
            split_directions=split_directions,
            freq="12h",
        )

        # Should have correct num rows
        if split_directions:
            assert rts.shape[0] >= len(dates) * n * 2
        else:
            # date span * num routes * num time chunks
            assert rts.shape[0] == len(dates) * n * 2

        # Should have correct columns
        expect_cols = {
            "datetime",
            "route_id",
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "service_distance",
            "service_duration",
            "service_speed",
        }
        if split_directions:
            expect_cols |= {"direction_id"}
        assert set(rts.columns) == expect_cols

        # Each route have a correct num_trip_starts
        if not split_directions:
            for route_id, rsr in rs.groupby("route_id"):
                get = rts.loc[
                    lambda x: x["route_id"] == route_id, "num_trip_starts"
                ].sum()
                expect = rsr["num_trip_starts"].sum()
                assert get == pytest.approx(expect)

        # Empty dates should yield empty DataFrame
        rts = gkr.compute_route_time_series(
            feed, ["19990101"], trip_stats, split_directions=split_directions
        )
        assert rts.empty
