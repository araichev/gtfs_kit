import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from gtfs_kit import cleaners as gkc
from gtfs_kit import constants as cn
from gtfs_kit import helpers as hp

from .context import gtfs_kit, sample


def test_clean_column_names():
    f = sample.routes.copy()
    g = gkc.clean_column_names(f)
    assert_frame_equal(f, g)

    f = sample.routes.copy()
    f[" route_id  "] = f["route_id"].copy()
    del f["route_id"]
    g = gkc.clean_column_names(f)
    assert "route_id" in g.columns
    assert " route_id  " not in g.columns


def test_clean_ids():
    f1 = sample.copy()
    f1.routes.loc[0, "route_id"] = "  ho   ho ho "
    f2 = gkc.clean_ids(f1)
    expect_rid = "ho_ho_ho"
    assert f2.routes.loc[0, "route_id"] == expect_rid

    f3 = gkc.clean_ids(f2)
    assert f3 == f2


def test_extend_id():
    f1 = sample.copy()
    assert len(f1.trips.set_index("route_id").loc["AAMV"]) == 4

    f2 = gkc.extend_id(f1, "route_id", "prefix_")
    assert f2.routes.route_id.str.startswith("prefix_").all()
    assert len(f2.trips.set_index("route_id").loc["prefix_AAMV"]) == 4

    f3 = gkc.extend_id(f2, "route_id", "_suffix", prefix=False)
    assert f3.routes.route_id.str.endswith("_suffix").all()
    assert len(f3.trips.set_index("route_id").loc["prefix_AAMV_suffix"]) == 4

    with pytest.raises(ValueError):
        gkc.extend_id(f2, "direction_id", "_suffix", prefix=False)


def test_prefix_feed_ids():
    prefix = "0_"
    f1 = sample.copy()
    f2 = gkc.prefix_feed_ids(f1, prefix)
    assert f2.routes.route_id.str.startswith(prefix).all()
    assert f2.agency.agency_id.str.startswith(prefix).all()


def test_clean_times():
    f1 = sample.copy()
    f1.stop_times.loc[0, "departure_time"] = "7:00:00"
    f1.frequencies.loc[0, "start_time"] = "7:00:00"
    f2 = gkc.clean_times(f1)
    assert f2.stop_times.loc[0, "departure_time"] == "07:00:00"
    assert f2.frequencies.loc[0, "start_time"] == "07:00:00"


def test_clean_route_short_names():
    f1 = sample.copy()

    # Should have no effect on a fine feed
    f2 = gkc.clean_route_short_names(f1)
    assert_series_equal(f2.routes["route_short_name"], f1.routes["route_short_name"])

    # Make route short name duplicates
    f1.routes.loc[1:5, "route_short_name"] = np.nan
    f1.routes.loc[6:, "route_short_name"] = "  he llo  "
    f2 = gkc.clean_route_short_names(f1)
    # Should have unique route short names
    assert f2.routes["route_short_name"].nunique() == f2.routes.shape[0]
    # NaNs should be replaced by n/a and route IDs
    expect_rsns = ("n/a-" + sample.routes.iloc[1:5]["route_id"]).tolist()
    assert f2.routes.iloc[1:5]["route_short_name"].values.tolist() == expect_rsns
    # Should have names without leading or trailing whitespace
    assert not f2.routes["route_short_name"].str.startswith(" ").any()
    assert not f2.routes["route_short_name"].str.endswith(" ").any()


def test_clean_parent_stations():
    fd = sample.copy()
    stops = fd.stops.copy()

    # setup: 'S1' is a stop (type 0), 'S2' is a station (type 1)
    # 'S3' is a stop pointing to a non-existent ID
    stops.loc[stops.index[0], ['stop_id', 'location_type', 'parent_station']] = ['S1', 0, 'S2']
    stops.loc[stops.index[1], ['stop_id', 'location_type']] = ['S2', 1]
    stops.loc[stops.index[2], ['stop_id', 'parent_station']] = ['S3', 'GHOST_STATION']

    fd.stops = stops
    fd_clean = gkc.clean_parent_stations(fd)

    # S1 -> S2 is valid, should remain
    assert fd_clean.stops[fd_clean.stops.stop_id == 'S1'].parent_station.iloc[0] == 'S2'
    # S3 -> GHOST_STATION is invalid, should be NA
    assert pd.isna(fd_clean.stops[fd_clean.stops.stop_id == 'S3'].parent_station.iloc[0])
    # S2 -> S1 not pointing to a non-station type
    stops.loc[stops.stop_id == 'S2', 'parent_station'] = 'S1'
    fd.stops = stops
    fd_clean_type = gkc.clean_parent_stations(fd)
    assert pd.isna(fd_clean_type.stops[fd_clean_type.stops.stop_id == 'S2'].parent_station.iloc[0])


def test_drop_zombies():
    # Should have no effect on sample feed
    f1 = sample.copy()
    f2 = gkc.drop_zombies(f1)
    assert_frame_equal(f2.routes, f1.routes)

    # Should drop stops with no stop times
    f1 = sample.copy()
    f1.stops["location_type"] = np.nan
    stop_id = f1.stops.stop_id.iat[0]
    st = f1.stop_times.copy()
    st = st.loc[lambda x: x.stop_id != stop_id]
    f1.stop_times = st
    f2 = gkc.drop_zombies(f1)
    assert stop_id not in f2.stops.stop_id.values

    f2 = gkc.drop_zombies(f1)
    assert_frame_equal(f2.routes, f1.routes)

    # Create undefined parent stations
    f1 = sample.copy()
    f1.stops["parent_station"] = "bingo"
    f2 = gkc.drop_zombies(f1)
    assert f2.stops.parent_station.isna().all()

    # Create all zombie trips for one route
    rid = f1.routes["route_id"].iat[0]
    cond = f1.trips["route_id"] == rid
    f1.trips.loc[cond, "trip_id"] = "hoopla"
    f2 = gkc.drop_zombies(f1)
    # Trips should be gone
    assert "hoopla" not in f2.trips["trip_id"]
    # Route should be gone
    assert rid not in f2.routes["route_id"]


def test_build_aggregate_routes_dict():
    routes = sample.routes.copy()
    # Equalize all route short names
    routes["route_short_name"] = "bingo"
    nid_by_oid = gkc.build_aggregate_routes_dict(routes, route_id_prefix="bongo_")
    assert set(nid_by_oid.values()) == {"bongo_0"}


def test_aggregate_routes():
    feed1 = sample.copy()
    # Equalize all route short names
    feed1.routes["route_short_name"] = "bingo"
    feed2 = gkc.aggregate_routes(feed1)

    # feed2 should have only one route ID
    assert feed2.routes.shape[0] == 1

    # Feeds should have same trip DataFrames excluding route IDs
    feed1.trips["route_id"] = feed2.trips["route_id"]
    assert hp.almost_equal(feed1.trips, feed2.trips)

    # Feeds should have same fare rules DataFrames excluding route IDs
    feed1.fare_rules["route_id"] = feed2.fare_rules["route_id"]
    assert hp.almost_equal(feed1.fare_rules, feed2.fare_rules)

    # Feeds should have equal attributes excluding routes, trips, and fare rules
    # DataFrames
    feed2.routes = feed1.routes
    feed2.trips = feed1.trips
    feed2.fare_rules = feed1.fare_rules
    assert feed1 == feed2


def test_build_aggregate_stops_dict():
    stops = sample.stops.copy()
    # Equalize all stop codes
    stops["stop_code"] = "bingo"
    nid_by_oid = gkc.build_aggregate_stops_dict(stops, stop_id_prefix="bongo_")
    assert set(nid_by_oid.values()) == {"bongo_0"}


def test_aggregate_stops():
    feed1 = sample.copy()
    # Equalize all stop codes
    feed1.stops["stop_code"] = "bingo"
    feed2 = gkc.aggregate_stops(feed1)

    # feed2 should have only one stop ID
    assert feed2.stops.shape[0] == 1

    # Feeds should have same stop times, excluding stop IDs
    feed1.stop_times["stop_id"] = feed2.stop_times.stop_id
    assert hp.almost_equal(feed1.stop_times, feed2.stop_times)

    # Feeds should have equal attributes excluding
    # stops stop times DataFrames
    feed2.stops = feed1.stops
    feed2.stop_times = feed1.stop_times
    assert feed1 == feed2


def test_clean():
    f1 = sample.copy()
    rid = f1.routes["route_id"].iat[0]
    f1.routes.loc[0, "route_id"] = " " + rid + "   "
    f2 = gkc.clean(f1)
    assert f2.routes.loc[0, "route_id"] == rid
    assert_frame_equal(f2.trips, sample.trips)


def test_drop_invalid_columns():
    f1 = sample.copy()
    f1.routes["bingo"] = "bongo"
    f1.trips["wingo"] = "wongo"
    f2 = gkc.drop_invalid_columns(f1)
    print(f2.routes)
    assert f2 == sample
