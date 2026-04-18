import pytest
import numpy as np

from gtfs_kit import cleaners as gkc
from gtfs_kit import merge as gmg

from .context import gtfs_kit, sample


def test_remap_ids():
    fd = sample.copy()
    sp = fd.stops.copy()

    # make a dummy parent station for one of the stops
    sp['parent_station'] = ["AMV"] + [""]*(len(sp)-1)
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
    assert conflicts == [{
        'stop_id_0': 'AMV', 'stop_id_1': 'AMV', 'match_key': 'AMV',
        'stop_name_0': 'Amargosa Valley (Demo)',
        'stop_name_1': 'Amargosa Valley (Demo)'
    }]
    assert len(f1.stops) == 8


def test_merge_similar_routes():
    f0 = sample.copy()
    f1 = sample.copy()

    f0.routes = f0.routes[f0.routes.route_short_name == '10'].copy()
    f1.routes = f1.routes[f1.routes.route_short_name == '10'].copy()

    f0_res, f1_res, conflicts = f0.merge_similar_routes(f1)
    # did it find the match?
    assert len(conflicts) == 1
    assert conflicts == [{
        'route_id_0': 'AB', 'route_id_1': 'AB',
        'route_short_name_0': '10', 'route_type_0': 3
    }]
    # was the route removed from f1.routes?
    assert f1_res.routes.empty
