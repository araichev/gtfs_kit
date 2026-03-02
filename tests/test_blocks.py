import itertools

import folium as fl
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from gtfs_kit import constants as cs
from gtfs_kit import blocks as gkb

from .context import (
    DATA_DIR,
    cairns,
    cairns_dates,
    cairns_shapeless,
    cairns_trip_stats,
    gtfs_kit,
    sample
)

sample_week = sample.get_first_week()
sample_trip_stats = sample.compute_trip_stats()

def test_get_blocks():
    feed = sample.copy()
    date = sample_week[1]
    f = gkb.get_blocks(feed, date)
    # Should have the correct shape
    assert f.shape[0] <= feed.trips[['block_id', 'service_id']].drop_duplicates().shape[0]

    # Test geo options
    feed = cairns.copy()
    g = gkb.get_blocks(feed, as_gdf=True, use_utm=True)
    assert isinstance(g, gpd.GeoDataFrame)
    assert g.crs != cs.WGS84

    with pytest.raises(ValueError):
        gkb.get_blocks(cairns_shapeless, as_gdf=True)

def test_compute_block_stats_0():
    feed = sample
    ts = sample_trip_stats.copy()

    expect_cols = set([
        "block_id",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "num_stop_patterns",
        "start_time",  # HH:MM:SS
        "end_time",  # HH:MM:SS
        "max_headway",  # minutes
        "min_headway",  # minutes
        "mean_headway",  # minutes
        "peak_num_trips",
        "peak_start_time",  # HH:MM:SS
        "peak_end_time",  # HH:MM:SS
        "service_distance",
        "service_duration",  # hours
        "service_speed",
        "mean_trip_distance",
        "mean_trip_duration",
    ])

    bs = gkb.compute_block_stats_0(ts)
    assert set(bs.columns) == expect_cols

    # test error w no valid block_id
    feed = cairns.copy()
    # note - not reading saved test trip_stats bc does not contain block_id
    ts1 = cairns.compute_trip_stats()
    with pytest.raises(ValueError):
        gkb.compute_block_stats_0(ts1)
    

def test_compute_block_stats():
    feed = sample
    ts = sample_trip_stats.copy()
    expect_cols = ["date"] + \
    [
        "block_id",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "num_stop_patterns",
        "start_time",  # HH:MM:SS
        "end_time",  # HH:MM:SS
        "max_headway",  # minutes
        "min_headway",  # minutes
        "mean_headway",  # minutes
        "peak_num_trips",
        "peak_start_time",  # HH:MM:SS
        "peak_end_time",  # HH:MM:SS
        "service_distance",
        "service_duration",  # hours
        "service_speed",
        "mean_trip_distance",
        "mean_trip_duration",
    ]
    expect_cols = set(expect_cols)
    bs = feed.compute_block_stats(sample_week[2])
    assert set(bs.columns) == expect_cols
    

def test_compute_block_time_series_0():
    feed = sample
    ts = sample_trip_stats.copy()

    expect_cols=set([
        "datetime",
        "block_id",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "service_distance",
        "service_duration",
        "service_speed",
    ])
    bts = gkb.compute_block_time_series_0(ts, sample_week[2])
    assert set(bts.columns) == expect_cols

    expect_cols2=list(expect_cols)
    expect_cols2=set(expect_cols2 + ['is_active'])
    bts2 = gkb.compute_block_time_series_0(ts, sample_week[2], active_blocks=True)
    assert set(bts2.columns) == expect_cols2

def test_compute_block_time_series():
    feed = sample
    ts = sample_trip_stats.copy()
    
    expect_cols=set([
        "datetime",
        "block_id",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "service_distance",
        "service_duration",
        "service_speed",
    ])
    bts = feed.compute_block_time_series(sample_week, trip_stats=ts)
    assert set(bts.columns) == expect_cols

    expect_cols2=list(expect_cols)
    expect_cols2=set(expect_cols2 + ['is_active'])
    bts2 = feed.compute_block_time_series(sample_week, trip_stats=ts, active_blocks=True)
    assert set(bts2.columns) == expect_cols2