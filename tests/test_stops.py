import itertools

import folium as fl
import geopandas as gp
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gtfs_kit import calendar as gkc
from gtfs_kit import stops as gks

from .context import DATA_DIR, cairns, cairns_dates, gtfs_kit

sample = gtfs_kit.read_feed(DATA_DIR / "sample_gtfs_2.zip", dist_units="km")


def test_get_stops():
    feed = cairns.copy()
    date = cairns_dates[0]
    trip_ids = feed.trips.trip_id.loc[:1]
    route_ids = feed.routes.route_id.loc[:1]
    frames = [
        gks.get_stops(feed),
        gks.get_stops(feed, date=date),
        gks.get_stops(feed, trip_ids=trip_ids),
        gks.get_stops(feed, route_ids=route_ids),
        gks.get_stops(feed, date=date, trip_ids=trip_ids),
        gks.get_stops(feed, date=date, route_ids=route_ids),
        gks.get_stops(feed, date=date, trip_ids=trip_ids, route_ids=route_ids),
    ]
    for f in frames:
        # Should be a data frame
        assert isinstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        assert f.shape[0] <= feed.stops.shape[0]
        assert f.shape[1] == feed.stops.shape[1]
        # Should have correct columns
        set(f.columns) == set(feed.stops.columns)
    # Number of rows should be reasonable
    assert frames[0].shape[0] <= frames[1].shape[0]
    assert frames[2].shape[0] <= frames[4].shape[0]
    assert frames[4].shape == frames[6].shape

    g = gks.get_stops(feed, as_gdf=True)
    assert isinstance(g, gpd.GeoDataFrame)
    assert g.crs == "epsg:4326"

    g = gks.get_stops(feed, as_gdf=True, use_utm=True)
    assert isinstance(g, gpd.GeoDataFrame)
    assert g.crs != "epsg:4326"


def test_compute_stop_activity():
    feed = cairns.copy()
    dates = gkc.get_first_week(feed)
    stop_activity = gks.compute_stop_activity(feed, dates)
    # Should be a data frame
    assert isinstance(stop_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert stop_activity.shape[0] == feed.stops.shape[0]
    assert stop_activity.shape[1] == len(dates) + 1
    # Date columns should contain only zeros and ones
    assert set(stop_activity[dates].values.flatten()) == {0, 1}


def test_build_stop_timetable():
    feed = sample.copy()
    stop_id = feed.stops["stop_id"].values[0]
    dates = feed.get_first_week()[:2]
    f = gks.build_stop_timetable(feed, stop_id, dates)

    # Should have the correct columns
    expect_cols = set(feed.trips.columns) | set(feed.stop_times.columns) | set(["date"])
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert f.date.unique().tolist() == dates

    # Empty check
    f = gks.build_stop_timetable(feed, stop_id, [])
    assert f.empty
    assert set(f.columns) == expect_cols


def test_geometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = gks.geometrize_stops(stops, use_utm=True)
    # Should be a GeoDataFrame
    assert isinstance(geo_stops, gp.GeoDataFrame)
    # Should have the correct shape
    assert geo_stops.shape[0] == stops.shape[0]
    assert geo_stops.shape[1] == stops.shape[1] - 1
    # Should have the correct columns
    expect_cols = set(list(stops.columns) + ["geometry"]) - set(["stop_lon", "stop_lat"])
    assert set(geo_stops.columns) == expect_cols


def test_ungeometrize_stops():
    stops = cairns.stops.copy()
    geo_stops = gks.geometrize_stops(stops)
    stops2 = gks.ungeometrize_stops(geo_stops)
    # Test columns are correct
    assert set(stops2.columns) == set(stops.columns)
    # Data frames should be equal after sorting columns
    cols = sorted(stops.columns)
    assert_frame_equal(stops2[cols], stops[cols])


def test_build_geometry_by_stop():
    d = gks.build_geometry_by_stop(cairns)
    assert isinstance(d, dict)
    assert len(d) == cairns.stops.stop_id.nunique()


def test_stops_to_geojson():
    feed = cairns.copy()
    stop_ids = feed.stops["stop_id"].tolist()[:2]
    gj = gks.stops_to_geojson(feed, stop_ids)
    assert isinstance(gj, dict)
    assert len(gj["features"]) == len(stop_ids)
    gj = gks.stops_to_geojson(feed, ["bingo"])
    assert not gj["features"]


def test_get_stops_in_area():
    feed = cairns.copy()
    area = gp.read_file(DATA_DIR / "cairns_square_stop_750070.geojson")
    stops = gks.get_stops_in_area(feed, area)
    expect_stop_ids = ["750070"]
    assert stops["stop_id"].values == expect_stop_ids


def test_map_stops():
    feed = cairns.copy()
    m = gks.map_stops(feed, feed.stops.stop_id.iloc[:5])
    assert isinstance(m, fl.Map)


def test_compute_stop_stats_0():
    feed1 = cairns.copy()
    feed2 = cairns.copy()
    feed2.trips.direction_id = pd.NA
    stop_times = feed1.stop_times.iloc[:250]

    for feed, split_directions in itertools.product([feed1, feed2], [True, False]):
        if split_directions and feed.trips.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                gks.compute_stop_stats_0(
                    stop_times, feed.trips, split_directions=split_directions
                )
            continue

        stops_stats = gks.compute_stop_stats_0(
            stop_times, feed.trips, split_directions=split_directions
        )
        # Should be a data frame
        assert isinstance(stops_stats, pd.core.frame.DataFrame)
        # Should contain the correct columns
        expect_cols = {
            "stop_id",
            "num_routes",
            "num_trips",
            "max_headway",
            "min_headway",
            "mean_headway",
            "start_time",
            "end_time",
        }
        if split_directions:
            expect_cols |= {"direction_id"}
        assert set(stops_stats.columns) == expect_cols

        # Should contain the correct stops
        expect_stops = set(stop_times["stop_id"].values)
        get_stops = set(stops_stats["stop_id"].values)
        assert get_stops == expect_stops

    # Empty check
    stats = gks.compute_stop_stats_0(feed.stop_times, pd.DataFrame())
    assert stats.empty


def test_compute_stop_stats():
    dates = cairns_dates
    feed = cairns.copy()
    n = 3
    sids = feed.stops.loc[:n, "stop_id"]
    for split_directions in [True, False]:
        f = gks.compute_stop_stats(
            feed, dates + ["19990101"], stop_ids=sids, split_directions=split_directions
        )

        # Should be a data frame
        assert isinstance(f, pd.core.frame.DataFrame)

        # Should contain the correct stops
        get = set(f["stop_id"].values)
        g = gks.get_stops(feed, date=dates[0]).loc[lambda x: x["stop_id"].isin(sids)]
        expect = set(g["stop_id"].values)
        assert get == expect

        # Should contain the correct columns
        expect_cols = {
            "date",
            "stop_id",
            "num_routes",
            "num_trips",
            "max_headway",
            "min_headway",
            "mean_headway",
            "start_time",
            "end_time",
        }
        if split_directions:
            expect_cols.add("direction_id")

        assert set(f.columns) == expect_cols

        # Should have correct dates
        set(f["date"].tolist()) == set(cairns_dates)

        # Non-feed dates should yield empty DataFrame
        f = gks.compute_stop_stats(feed, ["19990101"], split_directions=split_directions)
        assert f.empty


def test_compute_stop_time_series_0():
    feed1 = cairns.copy()
    feed2 = cairns.copy()
    feed2.trips["direction_id"] = pd.NA
    stop_times = feed1.stop_times.iloc[:250]
    nstops = stop_times["stop_id"].nunique()
    for feed, split_directions in itertools.product([feed1, feed2], [True, False]):
        if split_directions and feed.trips.direction_id.isnull().all():
            # Should raise an error
            with pytest.raises(ValueError):
                gks.compute_stop_time_series_0(
                    stop_times, feed.trips, split_directions=split_directions
                )
            continue

        ss = gks.compute_stop_stats_0(
            stop_times, feed.trips, split_directions=split_directions
        )
        sts = gks.compute_stop_time_series_0(
            stop_times, feed.trips, freq="12h", split_directions=split_directions
        )

        # Should have correct num rows and column names
        if split_directions:
            expect_cols = {"datetime", "stop_id", "direction_id", "num_trips"}
            assert sts.shape[0] <= nstops * 2
        else:
            expect_cols = {"datetime", "stop_id", "num_trips"}
            assert sts.shape[0] == nstops * 2
        assert set(sts.columns) == expect_cols

        # Each stop should have a correct total trip count
        if not split_directions:
            for stop_id, ssg in ss.groupby("stop_id"):
                get = sts.loc[lambda x: x["stop_id"] == stop_id]["num_trips"].sum()
                expect = ssg["num_trips"].sum()
                assert get == expect

    # Empty check
    stops_ts = gks.compute_stop_time_series_0(
        feed.stop_times, pd.DataFrame(), freq="1h", split_directions=split_directions
    )
    assert stops_ts.empty


def test_compute_stop_time_series():
    feed = cairns.copy()
    dates = cairns_dates
    n = 3
    stop_ids = feed.stops.loc[:n, "stop_id"]

    for split_directions in [True, False]:
        ss = gks.compute_stop_stats(
            feed, dates, stop_ids=stop_ids, split_directions=split_directions
        )
        sts = gks.compute_stop_time_series(
            feed,
            dates + ["20010101"],
            stop_ids=stop_ids,
            freq="12h",
            split_directions=split_directions,
        )

        # Should have correct num rows and column names
        k = len(stop_ids) * len(dates) * 2
        if split_directions:
            expect_cols = {"datetime", "stop_id", "direction_id", "num_trips"}
            assert sts.shape[0] <= k
        else:
            expect_cols = {"datetime", "stop_id", "num_trips"}
            assert sts.shape[0] == k
        assert set(sts.columns) == expect_cols

        # Each stop should have a correct total trip count
        if not split_directions:
            for stop_id, ssg in ss.groupby("stop_id"):
                get = sts.loc[lambda x: x["stop_id"] == stop_id]["num_trips"].sum()
                expect = ssg["num_trips"].sum()
                assert get == expect

        # Dates should be correct
        set(sts["datetime"].dt.strftime("%Y%m%d").values) == set(dates)

    # Empty check
    stops_ts = gks.compute_stop_time_series(
        feed, dates=["19990101"], split_directions=split_directions
    )
    assert stops_ts.empty
