"""
Functions about blocks.
"""
from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Callable, List

import datetime as dt
import numpy as np
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
    as_gdf: bool = False,
    use_utm: bool = False,
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
    as_gdf : bool | None
        If True and``feed.shapes`` is not ``None``, then return a GeoDataFrame with a geometry column of (Multi)LineStrings, each of which represents the corresponding block's union of trip shapes.
    use_utm : bool | None    
        The GeoDataFrame will have a local UTM CRS if ``use_utm``; otherwise it will have CRS WGS84.

    Returns
    -------
    pd.DataFrame
        A DataFrame with block IDs as index and columns
        - start_dt: datetime object representing the start time of the block
        - end_dt: datetime object representing the end time of the block

    Notes
    -----
    


    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.
    - Raise a KeyError if feed.trips does not contain a block_id column.
        
    """
    # Make sure block_id is in trip columns.
    if 'block_id' not in feed.trips.columns:
        raise KeyError("Trips feed must contain a block_id column to compute block stats.")

    final_cols = ['start_dt', 'end_dt', 'duration', 'distance', 'trip_ids', 'route_ids', 'speed']
    groupby_cols = ['block_id']
    
    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()

    # Get relevant trips with connected block_id
    # Add block_ids to trip stats
    trips = feed.get_trips(date=date, as_gdf=as_gdf, use_utm=use_utm) 
    trips = trips.merge(trip_stats)

    # Attach datetimes to trips
    block_date = _datestr_to_dt(date)
    if block_date is None:
        return pd.DataFrame(columns=final_cols)

    for time_col, dt_col in [("start_time", "start_dt"), ("end_time", "end_dt")]:
        trips[dt_col] = trips[time_col].map(lambda x: _timestr_to_dt(x, block_date))

    # Find all blocks    
    f = trips.groupby(groupby_cols).agg(
        start_dt=("start_dt", "min"), 
        end_dt=("end_dt", "max"),
        duration=("duration", "sum"),
        distance=("distance", "sum"),
        trip_ids=("trip_id", "unique"),
        route_ids=("route_id", "unique"),
        ).reset_index()
    f["speed"] = f["distance"] / f["duration"]

    if as_gdf:
        # return geodatframe
        import shapely.ops as so
        import geopandas as gpd

        
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")
        else:
            final_cols = f.columns.tolist() + ["geometry"]

        def merge_lines(group):
            lines = [
                g
                for g in group["geometry"]
                if g and g.geom_type in ["LineString", "MultiLineString"]
            ]
            if not lines:
                return pd.Series({"geometry": None})
            return pd.Series({"geometry": so.linemerge(lines)})

        f = (
            trips
            .filter(groupby_cols + ["geometry"])
            .groupby(groupby_cols)
            .apply(merge_lines, include_groups=False)
            .reset_index()
            .merge(f, how="right")
            .pipe(gpd.GeoDataFrame)
            .set_crs(trips.crs)
            .filter(final_cols)
        )

    return f


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

    # Count block starts and ends by freq
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

    # Calculate active blocks by freq as cumsum
    # of difference between block starts and ends
    active_blocks = block_starts - block_ends.shift(fill_value=0)
    active_blocks = active_blocks.cumsum()
    active_blocks.name = 'active_blocks'

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
        block_stats = compute_block_stats(feed, d, trip_stats).set_index('block_id', drop=True)
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




def compute_block_stats_0(
    trip_stats: pd.DataFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    final_cols = [
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
    null_stats = pd.DataFrame(data=[], columns=final_cols)

    # Handle defunct case
    if trip_stats.empty:
        return null_stats

    # Remove defunct trips
    f = trip_stats.loc[lambda x: x["duration"] > 0].copy()

    # Convert trip start and end times to seconds to ease calculations below
    f[["start_time", "end_time"]] = f[["start_time", "end_time"]].map(
        hp.timestr_to_seconds
    )

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    def agg(group):
        # Take this group of all trips stats for a single block
        # and compute block-level stats.
        d = dict()
        d["num_trips"] = group.shape[0]
        d["num_trip_starts"] = group["start_time"].count()
        d["num_trip_ends"] = group.loc[group["end_time"] < 24 * 3600, "end_time"].count()
        d["num_stop_patterns"] = group["stop_pattern_name"].nunique()
        d["start_time"] = group["start_time"].min()
        d["end_time"] = group["end_time"].max()

        # Compute max and mean headway
        stimes = group["start_time"].values
        stimes = sorted(
            [stime for stime in stimes if headway_start <= stime <= headway_end]
        )
        headways = np.diff(stimes)
        if headways.size:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan

        # Compute peak num trips
        active_trips = hp.get_active_trips_df(group[["start_time", "end_time"]])
        times, counts = active_trips.index.values, active_trips.values
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]

        d["service_distance"] = group["distance"].sum()
        d["service_duration"] = group["duration"].sum()

        return pd.Series(d)

    
    g = (
        f.groupby('block_id')
        .apply(agg, include_groups=False)
        .reset_index()
    )

    # Compute a few more stats
    g["service_speed"] = (
        g["service_distance"]
        .div(g["service_duration"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    g["mean_trip_distance"] = g["service_distance"] / g["num_trips"]
    g["mean_trip_duration"] = g["service_duration"] / g["num_trips"]

    # Convert route times to time strings
    g[["start_time", "end_time", "peak_start_time", "peak_end_time"]] = g[
        ["start_time", "end_time", "peak_start_time", "peak_end_time"]
    ].map(lambda x: hp.seconds_to_timestr(x))

    return g.filter(final_cols)


def x_compute_block_stats(
    feed: "Feed",
    dates: list[str],
    trip_stats: pd.DataFrame | None = None,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False
) -> pd.DataFrame:

    '''
    null_stats = compute_route_stats_0(
        feed.trips.head(0), split_directions=split_directions
    )
    final_cols = ["date"] + list(null_stats.columns)
    null_stats = null_stats.assign(date=None).filter(final_cols)
    '''
    
    null_stats = pd.DateFrame()
    dates = feed.subset_dates(dates)
    
    # Handle defunct case
    if not dates:
        return null_stats

    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()
        
    

    
    return "Hello"