import pytest
import numpy as np

from gtfs_kit import cleaners as gkc
from gtfs_kit import merge as gmg

from .context import gtfs_kit, sample, cairns


def test_remap_ids():
    fd = sample.copy()
    sp = fd.stops.copy()

    # make a dummy parent station for one of the stops
    sp["parent_station"] = ["AMV"] + [""] * (len(sp) - 1)
    fd.stops = sp
    fd = fd.remap_ids({"AMV": "X"}, "stop_id")

    assert fd.stops.stop_id.str.contains("X").sum() == 1
    assert fd.stops.iloc[0].parent_station == "X"


def test_merge_similar_stops():
    fd = sample.copy()

    # check conflicts is created correctly
    # also check conflicts are removed
    f0, f1, conflicts = fd.merge_similar_stops(fd)
    assert len(conflicts) == len(f0.stops)
    assert len(f0.stops) == 9
    assert len(f1.stops) == 0

    # check stops are dropped on matching stop_id
    f0 = gkc.extend_id(fd, "stop_id", "0_")
    f0 = f0.remap_ids({"0_AMV": "AMV"}, "stop_id")
    f0, f1, conflicts = f0.merge_similar_stops(fd)
    assert conflicts == [
        {
            "stop_id_0": "AMV",
            "stop_id_1": "AMV",
            "match_key": "AMV",
            "stop_name_0": "Amargosa Valley (Demo)",
            "stop_name_1": "Amargosa Valley (Demo)",
        }
    ]
    assert len(f1.stops) == 8


def test_merge_stops_with_prefixes():
    f0 = sample.copy()
    f1 = sample.copy()

    # Give them different prefixes
    f0.stops['stop_id'] = '0_' + f0.stops['stop_id']
    f1.stops['stop_id'] = '1_' + f1.stops['stop_id']

    # Isolate one stop for testing: '0_AMV' and '1_AMV'
    f0.stops = f0.stops[f0.stops.stop_id == '0_AMV'].copy()
    f1.stops = f1.stops[f1.stops.stop_id == '1_AMV'].copy()

    f0_res, f1_res, conflicts = f0.merge_similar_stops(f1, prefix_0='0_', prefix_1='1_')

    assert len(conflicts) == 1
    assert f1_res.stops.empty


def test_merge_similar_routes():
    f0 = sample.copy()
    f1 = sample.copy()

    f0.routes = f0.routes[f0.routes.route_short_name == "10"].copy()
    f1.routes = f1.routes[f1.routes.route_short_name == "10"].copy()

    f0_res, f1_res, conflicts = f0.merge_similar_routes(f1)
    # did it find the match?
    assert len(conflicts) == 1
    assert conflicts == [
        {
            "route_id_0": "AB",
            "route_id_1": "AB",
            "route_short_name_0": "10",
            "route_type_0": 3,
        }
    ]
    # was the route removed from f1.routes?
    assert f1_res.routes.empty


def test_merge_similar_calendars():
    f0 = sample.copy()
    f1 = sample.copy()

    f0.calendar = f0.calendar.copy()
    f1.calendar = f1.calendar.copy()
    f1.calendar.saturday = [0, 1]
    f1 = f1.remap_ids({"FULLW": "NEW_NAME"}, "service_id")

    f0_res, f1_res, conflicts = f0.merge_similar_calendars(f1)

    assert len(conflicts) == 1
    assert len(f1_res.calendar) == 1


def test_concatenate_feeds():
    f0 = sample.copy()
    f1 = sample.copy()

    fd_merged = gmg.concatenate_feeds(f0, f1)
    assert len(fd_merged.stops) == len(f0.stops) + len(f1.stops)
    assert len(fd_merged.routes) == len(f0.routes) + len(f1.routes)

    # test missing table handling
    f0.calendar = None
    fd_partial = gmg.concatenate_feeds(f0, f1)
    assert fd_partial.calendar is not None
    assert len(fd_partial.calendar) == len(f1.calendar)


def test_merge_feeds():
    f0 = cairns.copy().restrict_to_routes(["110N-423"])
    f1 = cairns.copy().restrict_to_routes(["110N-423", "121-423"])

    # test merging two overlapping feeds
    f2, conflicts = f0.merge_feeds(f1)
    # check stops
    assert len(f2.stops) == len(f1.stops)
    assert len(conflicts["stop_id"]) == len(f0.stops)
    # check routes
    assert len(f2.routes) == len(f1.routes)
    assert len(conflicts["routes"]) == 1
    # check calendars
    assert len(f2.calendar) == len(f1.calendar)
    assert len(conflicts["calendars"]) == 2

