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
        'stop_id_0': 'AMV', 'stop_id_1': 'AMV', 'base_id': 'AMV',
        'stop_name_0': 'Amargosa Valley (Demo)',
        'stop_name_1': 'Amargosa Valley (Demo)'
    }]
    assert len(f1.stops) == 8


