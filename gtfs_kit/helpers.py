"""
Functions useful across modules.
"""

from __future__ import annotations

import copy
import datetime as dt
import functools as ft
import math
from bisect import bisect_left, bisect_right
from functools import cmp_to_key
from typing import Callable, Literal

import json2html as j2h
import numpy as np
import pandas as pd
import shapely.geometry as sg

from . import constants as cs


def datestr_to_date(x: str | None, format_str: str = "%Y%m%d") -> dt.date | None:
    """
    Convert a date string to a datetime.date.
    Return ``None`` if ``x is None``.
    """
    if x is None:
        return None
    return dt.datetime.strptime(x, format_str).date()


def date_to_datestr(x: dt.date | None, format_str: str = "%Y%m%d") -> str | None:
    """
    Convert a datetime.date to a formatted string.
    Return ``None`` if ``x is None``.
    """
    if x is None:
        return None
    return x.strftime(format_str)


def timestr_to_seconds(x: str, *, mod24: bool = False) -> int | np.nan:
    """
    Given an HH:MM:SS time string ``x``, return the number of seconds
    past midnight that it represents.
    In keeping with GTFS standards, the hours entry may be greater than
    23.
    If ``mod24``, then return the number of seconds modulo ``24*3600``.
    Return ``np.nan`` in case of bad inputs.
    """
    try:
        hours, mins, seconds = x.split(":")
        result = int(hours) * 3600 + int(mins) * 60 + int(seconds)
        if mod24:
            result %= 24 * 3600
    except Exception:
        result = np.nan
    return result


def seconds_to_timestr(x: int, *, mod24: bool = False) -> str | np.nan:
    """
    The inverse of :func:`timestr_to_seconds`.
    If ``mod24``, then first take the number of seconds modulo ``24*3600``.
    Return NAN in case of bad inputs.
    """
    try:
        seconds = int(x)
        if mod24:
            seconds %= 24 * 3600
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        result = f"{hours:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        result = np.nan
    return result


def timestr_mod24(timestr: str) -> int | np.nan:
    """
    Given a GTFS HH:MM:SS time string, return a timestring in the same
    format but with the hours taken modulo 24.
    Return NAN in case of bad inputes
    """
    try:
        hours, mins, secs = [int(x) for x in timestr.split(":")]
        hours %= 24
        result = f"{hours:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        result = np.nan
    return result


def replace_date(f: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Given a table with a datetime object column called 'datetime' and given a
    YYYYMMDD date string, replace the datetime dates with the given date
    and return the resulting table.
    """
    d = datestr_to_date(date)
    return f.assign(
        datetime=lambda x: x["datetime"].map(
            lambda t: t.replace(year=d.year, month=d.month, day=d.day)
        )
    )


def get_segment_length(
    linestring: sg.LineString, p: sg.Point, q: sg.Point | None = None
) -> float:
    """
    Given a Shapely linestring and two Shapely points,
    project the points onto the linestring, and return the distance
    along the linestring between the two points.
    If ``q is None``, then return the distance from the start of the
    linestring to the projection of ``p``.
    The distance is measured in the native coordinates of the linestring.
    """
    # Get projected distances
    d_p = linestring.project(p)
    if q is not None:
        d_q = linestring.project(q)
        d = abs(d_p - d_q)
    else:
        d = d_p
    return d


def get_max_runs(x) -> np.array:
    """
    Given a list of numbers, return a NumPy array of pairs
    (start index, end index + 1) of the runs of max value.

    Example::

        >>> get_max_runs([7, 1, 2, 7, 7, 1, 2])
        array([[0, 1],
               [3, 5]])

    Assume x is not empty.
    Recipe comes from
    `Stack Overflow <http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array>`_.
    """
    # Get 0-1 array where 1 marks the max values of x
    x = np.array(x)
    m = np.max(x)
    y = (x == m) * 1
    # Bound y by zeros to detect runs properly
    bounded = np.hstack(([0], y, [0]))
    # Get 1 at run starts and -1 at run ends
    diffs = np.diff(bounded)
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]
    return np.array([run_starts, run_ends]).T
    # # Get lengths of runs and find index of longest
    # idx = np.argmax(run_ends - run_starts)
    # return run_starts[idx], run_ends[idx]


def get_peak_indices(times: list, counts: list) -> np.array:
    """
    Given an increasing list of times as seconds past midnight and a
    list of trip counts at those respective times,
    return a pair of indices i, j such that times[i] to times[j] is
    the first longest time period such that for all i <= x < j,
    counts[x] is the max of counts.
    Assume times and counts have the same nonzero length.

    Examples::

        >>> times = [0, 10, 20, 30, 31, 32, 40]
        >>> counts = [7, 1, 2, 7, 7, 1, 2]
        >>> get_peak_indices(times, counts)
        array([0, 1])

        >>> counts = [0, 0, 0]
        >>> times = [18000, 21600, 28800]
        >>> get_peak_indices(times, counts)
        array([0, 3])

    """
    max_runs = get_max_runs(counts)

    def get_duration(a):
        return times[a[1]] - times[a[0]]

    if len(max_runs) == 1:
        result = max_runs[0]
    else:
        index = np.argmax(np.apply_along_axis(get_duration, 1, max_runs))
        result = max_runs[index]

    return result


def is_metric(dist_units: str) -> bool:
    """
    Return True if the given distance units equals 'm' or 'km';
    otherwise return False.
    """
    return dist_units in ["m", "km"]


def get_convert_dist(dist_units_in: str, dist_units_out: str) -> Callable[[float], float]:
    """
    Return a function of the form

      distance in the units ``dist_units_in`` ->
      distance in the units ``dist_units_out``

    Only supports distance units in :const:`constants.DIST_UNITS`.
    """
    di, do = dist_units_in, dist_units_out
    DU = cs.DIST_UNITS
    if not (di in DU and do in DU):
        raise ValueError(f"Distance units must lie in {DU}")

    d = {
        "ft": {"ft": 1, "m": 0.3048, "mi": 1 / 5280, "km": 0.000_304_8},
        "m": {"ft": 1 / 0.3048, "m": 1, "mi": 1 / 1609.344, "km": 1 / 1000},
        "mi": {"ft": 5280, "m": 1609.344, "mi": 1, "km": 1.609_344},
        "km": {"ft": 1 / 0.000_304_8, "m": 1000, "mi": 1 / 1.609_344, "km": 1},
    }
    return lambda x: d[di][do] * x


def almost_equal(f: pd.DataFrame, g: pd.DataFrame) -> bool:
    """
    Return ``True`` if and only if the given DataFrames are equal after
    sorting their columns names, sorting their values, and
    reseting their indices.
    """
    if f.empty or g.empty:
        return f.equals(g)
    else:
        # Put in canonical order
        F = f.sort_index(axis=1).sort_values(list(f.columns)).reset_index(drop=True)
        G = g.sort_index(axis=1).sort_values(list(g.columns)).reset_index(drop=True)
        return F.equals(G)


def is_not_null(df: pd.DataFrame, col_name: str) -> bool:
    """
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    if (
        isinstance(df, pd.DataFrame)
        and col_name in df.columns
        and df[col_name].notnull().any()
    ):
        return True
    else:
        return False


def get_active_trips_df(trip_times: pd.DataFrame) -> pd.Series:
    """
    Count the number of trips in ``trip_times`` that are active
    at any given time.

    Assume ``trip_times`` contains the columns

    - start_time: start time of the trip in seconds past midnight
    - end_time: end time of the trip in seconds past midnight

    Return a Series whose index is times from midnight when trips
    start and end and whose values are the number of active trips for that time.
    """
    active_trips = (
        pd.concat(
            [
                pd.Series(1, trip_times.start_time),  # departed add 1
                pd.Series(-1, trip_times.end_time),  # arrived subtract 1
            ]
        )
        .groupby(level=0, sort=True)
        .sum()
        .cumsum()
        .ffill()
    )
    return active_trips


def combine_time_series(
    series_by_indicator: dict[str, pd.DataFrame],
    *,
    kind: Literal["route", "stop"],
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Combine a dict of wide time series (one DataFrame per indicator, columns are entities)
    into a single long-form time series with columns

    - ``'datetime'``
    - ``'route_id'`` or ``'stop_id'``: depending on ``kind``
    - ``'direction_id'``: present if and only if ``split_directions``
    - one column per indicator provided in `series_by_indicator`
    - ``'service_speed'``: if both ``service_distance`` and ``service_duration`` present

    If ``split_directions``, then assume the original time series contains data
    separated by trip direction; otherwise, assume not.
    The separation is indicated by a suffix ``'-0'`` (direction 0) or ``'-1'``
    (direction 1) in the route ID or stop ID column values.
    """
    if not series_by_indicator:
        return pd.DataFrame()

    # Validate indices and types
    indicators = list(series_by_indicator.keys())
    base_index: pd.DatetimeIndex | None = None
    for ind, df in series_by_indicator.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Indicator '{ind}' is not a DataFrame.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Indicator '{ind}' must have a DatetimeIndex.")
        if base_index is None:
            base_index = df.index
        elif not base_index.equals(df.index):
            raise ValueError(
                "All indicator DataFrames must share the same DatetimeIndex."
            )

    entity_col = "route_id" if kind == "route" else "stop_id"

    # Wide to long for each indicator
    long_frames: list[pd.DataFrame] = []
    for ind, wf in series_by_indicator.items():
        s = wf.stack(future_stack=True).rename(ind)  # (datetime, entity) -> value
        lf = s.reset_index()
        lf.columns = ["datetime", entity_col, ind]
        long_frames.append(lf)

    # Merge all indicators on (datetime, entity)
    f = ft.reduce(
        lambda a, b: pd.merge(a, b, on=["datetime", entity_col], how="outer"),
        long_frames,
    )

    # Optionally split direction from encoded IDs "<id>-<dir>"
    if split_directions:

        def _split(ent):
            if pd.isna(ent):
                return np.nan, np.nan
            parts = str(ent).rsplit("-", 1)
            if len(parts) != 2:
                return ent, np.nan
            eid, did = parts
            try:
                return eid, int(did)
            except Exception:
                return eid, np.nan

        split = f[entity_col].map(_split)
        f[entity_col] = split.map(lambda t: t[0])
        f["direction_id"] = split.map(lambda t: t[1])

    # Coerce numeric indicators and fill NaNs with 0 (speed handled later)
    numeric_cols = []
    for ind in indicators:
        if ind in f.columns:
            f[ind] = pd.to_numeric(f[ind], errors="coerce")
            numeric_cols.append(ind)
    if numeric_cols:
        f[numeric_cols] = f[numeric_cols].fillna(0)

    # Compute service_speed if possible
    if "service_distance" in f.columns and "service_duration" in f.columns:
        f["service_speed"] = (
            f["service_distance"]
            .div(f["service_duration"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    # Arrange columns
    cols0 = ["datetime", entity_col]
    if split_directions:
        cols0.append("direction_id")
    cols = cols0 + [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
        "service_speed",
    ]
    return f.filter(cols).sort_values(cols0, ignore_index=True)


def downsample(time_series: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Downsample the given stop, route,  or network time series,
    (outputs of :func:`.stops.compute_stop_time_series`,
    :func:`.routes.compute_route_time_series`, or
    :func:`.miscellany.compute_network_time_series`,
    respectively) to the given frequency (Pandas frequency string, e.g. '15Min').

    Return the given time series unchanged if the given frequency is
    finer than the original frequency or the given frequncy is courser than a day.
    """
    import pandas.tseries.frequencies as pdf

    # Handle defunct cases
    if time_series.empty:
        return time_series

    f = time_series.assign(datetime=lambda x: pd.to_datetime(x["datetime"]))
    ifreq = pd.infer_freq(f["datetime"].unique()[:3])
    if ifreq is None:
        # Carry on, assuming everything will work
        pass
        # raise ValueError("Can't infer frequency of time series")
    elif pdf.to_offset(freq) <= pdf.to_offset(ifreq) or pdf.to_offset(
        freq
    ) > pdf.to_offset("24h"):
        return f

    # Handle generic case
    id_cols = list({"route_id", "stop_id", "direction_id", "route_type"} & set(f.columns))

    if not id_cols:
        # Network time series without route type
        f["tmp"] = "tmp"
        id_cols = ["tmp"]

    if "stop_id" in time_series.columns:
        is_stop_series = True
        indicators = ["num_trips"]
    else:
        # It's a route or network time series.
        is_stop_series = False
        indicators = [
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "service_distance",
            "service_duration",
            "service_speed",
        ]

    def agg_num_trips(g):
        """
        Num trips uses custom rule:
        last(num_trips in bin) + sum(num_trip_ends in all but the last row in the bin)
        """
        if g.empty:
            return np.nan
        return g["num_trips"].iloc[-1] + g["num_trip_ends"].iloc[:-1].sum(min_count=1)

    frames = []
    for key, group in f.groupby(id_cols, dropna=False):
        g = group.sort_values("datetime").set_index("datetime")
        # groupby + Grouper will not create empty bins
        gb = g.groupby(pd.Grouper(freq=freq))

        if is_stop_series:
            # Sum all numeric columns, preserving all-NaN groups (min_count=1)
            agg = (
                gb.sum(min_count=1)
                # Remove any extra dates inserted in between,
                # which will have all NAN values
                .dropna()
                .reset_index()
            )
        else:
            series = []
            for col in indicators:
                if col == "num_trips":
                    s = gb.apply(agg_num_trips)
                elif col != "service_speed":
                    s = gb[col].agg(lambda x: x.sum(min_count=1))
                series.append(s.rename(col))

            agg = (
                pd.concat(series, axis="columns", sort=False)
                # Remove any extra dates inserted in between,
                # which will have all NAN values
                .dropna()
                # Compute service speed now
                .assign(
                    service_speed=lambda x: x["service_distance"]
                    .div(x["service_duration"])
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
                # Bring back 'datetime' column
                .reset_index()
            )

        # Reattach ID columns
        if isinstance(key, tuple):
            for col, val in zip(id_cols, key):
                agg[col] = val
        else:
            agg[id_cols[0]] = key

        frames.append(agg)

    # Collate results
    cols0 = ["datetime"] + [x for x in id_cols if x != "tmp"]
    cols = cols0 + indicators
    return pd.concat(frames).filter(cols).sort_values(cols0, ignore_index=True)


def make_html(d: dict) -> str:
    """
    Convert the given dictionary into an HTML table (string) with
    two columns: keys of dictionary, values of dictionary.
    """
    return j2h.json2html.convert(
        json=d, table_attributes="class='table table-condensed table-hover'"
    )


def longest_subsequence(
    seq, mode="strictly", order="increasing", key=None, *, index=False
):
    """
    Return the longest increasing subsequence of `seq`.

    Parameters
    ----------
    seq : sequence object
      Can be any sequence, like `str`, `list`, `numpy.array`.
    mode : {'strict', 'strictly', 'weak', 'weakly'}, optional
      If set to 'strict', the subsequence will contain unique elements.
      Using 'weak' an element can be repeated many times.
      Modes ending in -ly serve as a convenience to use with `order` parameter,
      because `longest_sequence(seq, 'weakly', 'increasing')` reads better.
      The default is 'strict'.
    order : {'increasing', 'decreasing'}, optional
      By default return the longest increasing subsequence, but it is possible
      to return the longest decreasing sequence as well.
    key : function, optional
      Specifies a function of one argument that is used to extract a comparison
      key from each list element (e.g., `str.lower`, `lambda x: x[0]`).
      The default value is `None` (compare the elements directly).
    index : bool, optional
      If set to `True`, return the indices of the subsequence, otherwise return
      the elements. Default is `False`.

    Returns
    -------
    elements : list, optional
      A `list` of elements of the longest subsequence.
      Returned by default and when `index` is set to `False`.
    indices : list, optional
      A `list` of indices pointing to elements in the longest subsequence.
      Returned when `index` is set to `True`.

    Taken from `this Stack Overflow answer <https://stackoverflow.com/a/38337443>`_.
    """
    bisect = bisect_left if mode.startswith("strict") else bisect_right

    # compute keys for comparison just once
    rank = seq if key is None else map(key, seq)
    if order == "decreasing":
        rank = map(cmp_to_key(lambda x, y: 1 if x < y else 0 if x == y else -1), rank)
    rank = list(rank)

    if not rank:
        return []

    lastoflength = [0]  # end position of subsequence with given length
    predecessor = [None]  # penultimate element of l.i.s. ending at given position

    for i in range(1, len(seq)):
        # seq[i] can extend a subsequence that ends with a lesser (or equal) element
        j = bisect([rank[k] for k in lastoflength], rank[i])
        # update existing subsequence of length j or extend the longest
        try:
            lastoflength[j] = i
        except Exception:
            lastoflength.append(i)
        # remember element before seq[i] in the subsequence
        predecessor.append(lastoflength[j - 1] if j > 0 else None)

    # trace indices [p^n(i), ..., p(p(i)), p(i), i], where n=len(lastoflength)-1
    def trace(i):
        if i is not None:
            yield from trace(predecessor[i])
            yield i

    indices = trace(lastoflength[-1])

    return list(indices) if index else [seq[i] for i in indices]


def make_ids(n: int, prefix: str = "id_"):
    """
    Return a length ``n`` list of unique sequentially labelled strings for use as IDs.

    Example::

        >>> make_ids(11, prefix="s")
        ['s00', s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10']

    """
    if n < 1:
        result = []
    elif n == 1:
        result = [f"{prefix}0"]
    else:
        k = int(math.log10(n - 1)) + 1  # Number of digits for IDs
        result = [f"{prefix}{i:0{k}d}" for i in range(n)]

    return result
