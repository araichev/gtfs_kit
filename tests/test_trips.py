import pandas as pd
import numpy as np
import pytest
import geopandas as gpd
import folium as fl

from .context import (
    gtfs_kit,
    DATA_DIR,
    cairns,
    cairns_shapeless,
    cairns_dates,
    cairns_trip_stats,
)
from gtfs_kit import constants as cs
from gtfs_kit import trips as gkt
from gtfs_kit import calendar as gkc
from gtfs_kit import helpers as gkh
from gtfs_kit import stop_times as gks


def test_get_active_services():
    feed = cairns
    week = feed.get_first_week()
    s = gkt.get_active_services(feed, week[0])
    assert set() < set(s) < set(feed.trips["service_id"])

    # Should work with only one of `feed.calendar` and `feed.calendar_dates`
    for table in ["calendar", "calendar_dates"]:
        feed = cairns.copy()
        setattr(feed, table, None)
        week = feed.get_first_week()
        s = gkt.get_active_services(feed, week[0])
        assert set() < set(s) < set(feed.trips["service_id"])


def test_get_trips():
    feed = cairns.copy()
    date = cairns_dates[0]
    trips1 = gkt.get_trips(feed, date)
    # Should be a data frame
    assert isinstance(trips1, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips1.shape[0] <= feed.trips.shape[0]
    assert trips1.shape[1] == feed.trips.shape[1]
    # Should have correct columns
    assert set(trips1.columns) == set(feed.trips.columns)

    trips2 = gkt.get_trips(feed, date, "07:30:00")
    # Should be a data frame
    assert isinstance(trips2, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips2.shape[0] <= trips2.shape[0]
    assert trips2.shape[1] == trips1.shape[1]
    # Should have correct columns
    assert set(trips2.columns) == set(feed.trips.columns)

    feed = cairns.copy()
    g = gkt.get_trips(feed, as_gdf=True)
    assert g.crs == cs.WGS84
    assert g.shape[0] == feed.trips.shape[0]
    assert set(g.columns) == set(feed.trips.columns) | {"geometry"}

    with pytest.raises(ValueError):
        gkt.get_trips(cairns_shapeless, as_gdf=True)


def test_compute_trip_activity():
    feed = cairns.copy()
    dates = gkc.get_first_week(feed)
    ta = gkt.compute_trip_activity(feed, dates)
    # Should have the correct shape
    assert ta.shape == (feed.trips.shape[0], 1 + len(dates))
    # Date columns should contain only zeros and ones
    assert set(ta[dates].values.flatten()) == {0, 1}


def test_compute_busiest_date():
    feed = cairns.copy()
    dates = gkc.get_first_week(feed)[:1]
    date = gkt.compute_busiest_date(feed, dates + ["999"])
    # Busiest day should lie in first week
    assert date in dates


def test_name_stop_patterns():
    feed = cairns.copy()
    t = gkt.name_stop_patterns(feed)
    assert set(t.columns) == set(feed.trips.columns) | {"stop_pattern_name"}

    # Should still work without direction ID
    feed.trips = feed.trips.drop("direction_id", axis=1)
    t = gkt.name_stop_patterns(feed)
    assert set(t.columns) == set(feed.trips.columns) | {"stop_pattern_name"}


def test_compute_trip_stats():
    feed = cairns.copy()
    n = 3
    rids = feed.routes.route_id.loc[:n]
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids)

    # Should be a data frame with the correct number of rows
    trip_subset = feed.trips.loc[lambda x: x["route_id"].isin(rids)].copy()
    assert isinstance(trip_stats, pd.core.frame.DataFrame)
    assert trip_stats.shape[0] == trip_subset.shape[0]

    # Should contain the correct columns
    expect_cols = set(
        [
            "trip_id",
            "direction_id",
            "route_id",
            "route_short_name",
            "route_type",
            "shape_id",
            "stop_pattern_name",
            "num_stops",
            "start_time",
            "end_time",
            "start_stop_id",
            "end_stop_id",
            "distance",
            "duration",
            "speed",
            "is_loop",
        ]
    )
    assert set(trip_stats.columns) == expect_cols

    # Distance units should be correct
    d1 = trip_stats.distance.iat[0]  # km
    trip_stats_2 = gkt.compute_trip_stats(feed.convert_dist("ft"), route_ids=rids)
    d2 = trip_stats_2.distance.iat[0]  # mi
    f = gkh.get_convert_dist("km", "mi")
    assert np.allclose(f(d1), d2)

    # Shapeless feeds should have null entries for distance column
    feed = cairns_shapeless.copy()
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids)
    assert len(trip_stats["distance"].unique()) == 1
    assert np.isnan(trip_stats["distance"].unique()[0])

    # Should contain the correct trips
    get_trips = set(trip_stats["trip_id"].values)
    expect_trips = set(trip_subset["trip_id"].values)
    assert get_trips == expect_trips

    # Missing the optional ``direction_id`` column in ``feed.trips``
    # should give ``direction_id`` column in stats with all NaNs
    feed = cairns.copy()
    del feed.trips["direction_id"]
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids)
    assert set(trip_stats.columns) == expect_cols
    assert trip_stats.direction_id.isnull().all()

    # Missing the optional ``shape_id`` column in ``feed.trips``
    # should give ``shape_id`` column in stats with all NaNs
    feed = cairns.copy()
    del feed.trips["shape_id"]
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids)
    assert set(trip_stats.columns) == expect_cols
    assert trip_stats.shape_id.isnull().all()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_locate_trips():
    feed = cairns.copy()
    feed = gks.append_dist_to_stop_times(feed)
    date = cairns_dates[0]
    times = ["08:00:00"]
    f = gkt.locate_trips(feed, date, times)
    print(f.head().T)

    g = gkt.get_trips(feed, date, times[0])
    # Should have the correct number of rows
    assert f.shape[0] == g.shape[0]
    # Should have the correct columns
    expect_cols = set(
        [
            "route_id",
            "trip_id",
            "direction_id",
            "shape_id",
            "time",
            "rel_dist",
            "lon",
            "lat",
        ]
    )
    assert set(f.columns) == expect_cols

    # Missing feed.trips.shape_id should raise an error
    feed = cairns_shapeless.copy()
    del feed.trips["shape_id"]
    with pytest.raises(ValueError):
        gkt.locate_trips(feed, date, times)


def test_trips_to_geojson():
    feed = cairns.copy()
    trip_ids = feed.trips.trip_id.loc[:1]
    n = len(trip_ids)
    gj = gkt.trips_to_geojson(feed, trip_ids)
    assert len(gj["features"]) == n

    gj = gkt.trips_to_geojson(feed, trip_ids, include_stops=True)
    k = (
        feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)]
        .drop_duplicates(subset=["trip_id", "stop_id"])
        .shape[0]
    )
    assert len(gj["features"]) == n + k

    gj = gkt.trips_to_geojson(cairns, trip_ids=["bingo"])
    assert not gj["features"]

    with pytest.raises(ValueError):
        gkt.trips_to_geojson(cairns_shapeless)


def test_map_trips():
    feed = cairns.copy()
    tids = feed.trips["trip_id"].values[:2]
    m = gkt.map_trips(feed, tids, show_stops=True)
    assert isinstance(m, fl.Map)
