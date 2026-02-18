"""
Tests for blocks.
"""
import pytest
import pandas as pd
from datetime import date

from gtfs_kit import blocks as gkb

from .context import DATA_DIR, sample, cairns, cairns_dates, cairns_trip_stats

def test_block_time_series():
    feed = sample.copy()
    dates = feed.get_dates()
    bts = feed.compute_block_time_series(dates=dates[0:10], freq='1h')
    # should have same count of block starts and ends
    assert bts['block_starts'].sum() == bts['block_ends'].sum()

def test_block_stats():
    feed = sample.copy()
    dates = feed.get_dates()
    # should have same block_ids in block_stats for same dates
    trips = feed.get_trips(dates[1])
    bs = gkb.compute_block_stats(feed, dates[1])
    assert trips['block_id'].nunique() == bs.shape[0]

