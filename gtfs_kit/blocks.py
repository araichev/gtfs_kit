"""
Functions about blocks.
"""
from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Callable, List

import datetime as dt
import pandas as pd

from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def _datestr_to_dt(x: str | None, format_str: str = "%Y%m%d") -> dt.date | None:
    """
    Convert a date string to a datetime.date.
    Return ``None`` if ``x is None``.
    """
    if x is None:
        return None
    return dt.datetime.strptime(x, format_str).date()


def _timestr_to_dt(x: str, date: dt.date) -> dt.datetime:
    """
    Convert a time string and a date to a datetime object.
    """
    s = hp.timestr_to_seconds(x, mod24=True)
    return dt.datetime.combine(date, dt.time()) + dt.timedelta(seconds=s)


def compute_block_stats(
    feed: "Feed",
    date: str,
    trip_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute block start and end times for the given date.

    Parameters
    ----------
    feed : Feed
        A GTFS-Kit feed object.
    date : str
        A YYYYMMDD date string.
    trip_stats : pd.DataFrame | None
        A pre-computed trip stats DataFrame from ``feed.compute_trip_stats()``.
        If ``None``, then it will be computed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with block IDs as index and columns
        - start_dt: datetime object representing the start time of the block
        - end_dt: datetime object representing the end time of the block
    """
    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()

    # Get relevant trips with connected block_id
    # Add block_ids to trip stats
    ts = trip_stats.merge(feed.trips[["trip_id", "block_id"]])
    date_trips = feed.get_trips(date)["trip_id"]
    ts = ts[ts["trip_id"].isin(date_trips)]

    # Attach datetimes to trips
    block_date = _datestr_to_dt(date)
    if block_date is None:
        return pd.DataFrame(columns=["start_dt", "end_dt"])

    for time_col, dt_col in [("start_time", "start_dt"), ("end_time", "end_dt")]:
        ts[dt_col] = ts[time_col].map(lambda x: _timestr_to_dt(x, block_date))

    # Find stats for all blocks
    block_stats = ts.groupby("block_id").agg(
        start_dt=("start_dt", "min"), 
        end_dt=("end_dt", "max"),
        duration=("duration", "sum"),
        distance=("distance", "sum")
                 )
    block_stats["speed"] = block_stats["distance"] / block_stats["duration"]

    return block_stats


def _compute_block_time_series(
    block_stats: pd.DataFrame,
    freq: str = "h",
    block_list: list[str] | None = None,
    block_filt: Callable | None = None,
) -> pd.DataFrame:
    """
    Helper function for ``active_blocks_by_freq``.
    """
    if block_list:  # filter for list of blocks
        block_stats = block_stats[block_stats.index.isin(block_list)]
    if block_filt:  # filter blocks using function
        block_stats = block_stats[list(map(block_filt, block_stats.index))]

    # Return none if no blocks in block times
    if not block_stats.shape[0]:
        return pd.DataFrame(columns=["active_blocks", "block_starts", "block_ends"])

    # Create dt index to check for active blocks
    first_block = pd.Timestamp(block_stats["start_dt"].min())
    last_block = pd.Timestamp(block_stats["end_dt"].max())
    day_start = first_block.floor("1D")
    range_start = first_block.floor(freq)
    range_end = last_block.ceil(freq)
    dr = pd.date_range(start=range_start, end=range_end, freq=freq)
    full_dr = pd.date_range(start=day_start, end=range_end, freq=freq)

    # Count active blocks by hour
    active_blocks = []
    for s in dr:
        e = (s + pd.Timedelta(minutes=1)).ceil(freq)  # use (s)tart and (e)nd of freq
        ab = block_stats[(block_stats["start_dt"] < e) & (block_stats["end_dt"] >= s)]
        active_blocks.append(ab.shape[0])

    active_blocks = pd.Series(
        index=dr, data=active_blocks, name="active_blocks"
    ).reindex(full_dr, fill_value=0)

    # Count block starts and ends by hour
    block_starts = (
        pd.Series(index=block_stats["start_dt"], data=1, name="block_starts")
        .resample(freq)
        .count()
        .reindex(full_dr, fill_value=0)
    )

    block_ends = (
        pd.Series(index=block_stats["end_dt"], data=1, name="block_ends")
        .resample(freq)
        .count()
        .reindex(full_dr, fill_value=0)
    )

    return pd.concat([active_blocks, block_starts, block_ends], axis=1)


def compute_block_time_series(
    feed: "Feed",
    dates: list[str],
    freq: str,
    trip_stats: pd.DataFrame | None = None,
    block_list: list[str] | None = None,
    block_filt: Callable | None = None,
) -> pd.DataFrame:
    """
    Compute the number of active blocks, block starts, and block ends for a given
    list of dates, aggregated by a given frequency.

    Parameters
    ----------
    feed : Feed
        A GTFS-Kit feed object.
    dates : list[str]
        A list of YYYYMMDD date strings.
    freq : str
        A Pandas frequency string, e.g. 'h' for hourly.
    trip_stats : pd.DataFrame | None
        A pre-computed trip stats DataFrame from ``feed.compute_trip_stats()``.
        If ``None``, then it will be computed.
    block_list : list | None
        A list of block IDs to filter for.
    block_filt : Callable | None
        A function to filter block IDs. For example, ``lambda x: x.startswith('a')``

    Returns
    -------
    pd.DataFrame
        A DataFrame with a DatetimeIndex and columns
        - active_blocks: number of blocks active during the time period
        - block_starts: number of blocks that started during the time period
        - block_ends: number of blocks that ended during the time period
    """

    active_block_results = []
    for d in dates:
        block_stats = compute_block_stats(feed, d, trip_stats)
        active_blocks = _compute_block_time_series(
            block_stats, freq=freq, block_list=block_list, block_filt=block_filt
        )
        if active_blocks.shape[0]:
            active_block_results.append(active_blocks)
        else: 
            # create empty results if no blocks
            block_date = _datestr_to_dt(d)
            if block_date is None:
                continue
            empty_dt = pd.Timestamp(block_date)
            columns = ["active_blocks", "block_starts", "block_ends"]
            index = pd.date_range(
                start=empty_dt, end=empty_dt + pd.Timedelta(days=1), freq=freq
            )
            empty_df = pd.DataFrame(index=index, columns=columns, data=0)
            active_block_results.append(empty_df)

    if not active_block_results:
        return pd.DataFrame(columns=["active_blocks", "block_starts", "block_ends"])

    all_active_blocks = reduce(lambda a, b: a.add(b, fill_value=0), active_block_results)

    # Fill in any missing periods
    dr = pd.date_range(
        start=all_active_blocks.index.min(), end=all_active_blocks.index.max(), freq=freq
    )

    all_active_blocks = all_active_blocks.reindex(dr, fill_value=0)

    return all_active_blocks