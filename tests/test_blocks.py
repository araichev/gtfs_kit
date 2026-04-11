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
    assert "service_id" in f.columns

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
        "service_id",
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

    # test gtfs file without block ids.
    ts1 = cairns.compute_trip_stats()
    bs0 = gkb.compute_block_stats_0(ts1)
    assert bs0.shape[0] == 0
    assert set(bs0.columns) == expect_cols
    

def test_compute_block_stats():
    feed = sample
    ts = sample_trip_stats.copy()
    expect_cols = ["date"] + \
    [
        "block_id",
        "service_id",
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
        "service_id",
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
    
    # Value level check for is_active
    # Pick a block and check if it's active when it should be
    block_id = bts2.block_id.iloc[0]
    service_id = bts2.service_id.iloc[0]
    stats = gkb.compute_block_stats_0(ts.loc[lambda x: (x.block_id == block_id) & (x.service_id == service_id)])
    start_time = pd.to_datetime(f"{sample_week[2]} {stats.start_time.iloc[0]}")
    end_time = pd.to_datetime(f"{sample_week[2]} {stats.end_time.iloc[0]}")
    
    active_bins = bts2.loc[lambda x: (x.block_id == block_id) & (x.service_id == service_id) & x.is_active]
    assert not active_bins.empty
    for dt in active_bins.datetime:
        # A bin is active if its time interval overlaps with [start_time, end_time]
        # Our bins are 1 hour by default in compute_block_time_series_0
        bin_start = dt
        bin_end = dt + pd.Timedelta(hours=1)
        assert bin_start <= end_time and bin_end >= start_time

def test_compute_block_time_series():
    feed = sample
    ts = sample_trip_stats.copy()
    
    expect_cols=set([
        "datetime",
        "block_id",
        "service_id",
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

    # The number of period with active blocks in a time series should
    # always be more than or equual to the number of periods with
    # active trips.
    assert bts2['is_active'].sum() >= (bts2['num_trips'] > 0).sum()

def test_block_identity():
    # Test that blocks with same block_id but different service_id are treated as distinct
    feed = sample.copy()
    # Create a synthetic case: duplicate a block but with a different service_id
    trips = feed.trips.copy()
    block_id = trips.block_id.dropna().iloc[0]
    original_service_id = trips.loc[trips.block_id == block_id, "service_id"].iloc[0]
    
    new_service_id = "new_service"
    old_trip_ids = trips.loc[trips.block_id == block_id, "trip_id"].tolist()
    
    new_trips = trips.loc[trips.block_id == block_id].copy()
    new_trips["service_id"] = new_service_id
    new_trips["trip_id"] = new_trips["trip_id"] + "_new"
    
    feed.trips = pd.concat([trips, new_trips], ignore_index=True)
    
    # Also need to duplicate stop_times for the new trips
    new_stop_times = feed.stop_times.loc[feed.stop_times.trip_id.isin(old_trip_ids)].copy()
    new_stop_times["trip_id"] = new_stop_times["trip_id"] + "_new"
    feed.stop_times = pd.concat([feed.stop_times, new_stop_times], ignore_index=True)

    # Also need to add the new service to calendar/calendar_dates to make it "active"
    if feed.calendar is not None:
        # Find a row that is active on the date we want
        date = sample_week[0]
        weekday_str = pd.to_datetime(date).strftime("%A").lower()
        active_row = feed.calendar.loc[lambda x: (x.start_date <= date) & (x.end_date >= date) & (x[weekday_str] == 1)].iloc[:1].copy()
        active_row["service_id"] = new_service_id
        feed.calendar = pd.concat([feed.calendar, active_row], ignore_index=True)
    
    # Get blocks for a date where both services are active
    date = sample_week[0]
    blocks = feed.get_blocks(date)
    
    # Should have both (block_id, original_service_id) and (block_id, new_service_id)
    relevant_blocks = blocks.loc[blocks.block_id == block_id]
    assert len(relevant_blocks) == 2
    assert set(relevant_blocks.service_id) == {original_service_id, new_service_id}
    
    # Check stats
    ts = feed.compute_trip_stats()
    stats = feed.compute_block_stats([date], trip_stats=ts)
    relevant_stats = stats.loc[stats.block_id == block_id]
    assert len(relevant_stats) == 2
    assert set(relevant_stats.service_id) == {original_service_id, new_service_id}
    
    # Check time series
    bts = feed.compute_block_time_series([date], trip_stats=ts)
    relevant_bts = bts.loc[bts.block_id == block_id]
    # Should have entries for both services
    assert relevant_bts.service_id.nunique() == 2
