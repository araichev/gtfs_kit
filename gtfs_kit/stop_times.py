"""
Functions about stop times.
"""

from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import json

import pandas as pd
import numpy as np

from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_stop_times(feed: "Feed", date: str | None = None) -> pd.DataFrame:
    """
    Return ``feed.stop_times``.
    If a date (YYYYMMDD date string) is given, then subset the result to only those
    stop times with trips active on the date.
    """
    if date is None:
        f = feed.stop_times.copy()
    else:
        trip_ids = feed.get_trips(date).trip_id
        f = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    return f


def append_dist_to_stop_times(feed: "Feed") -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` column in
    ``feed.stop_times`` in terms of the distance units ``feed.dist_units``.
    Trips without shapes will have NaN distances.
    Return the resulting Feed.
    Uses ``feed.shapes``, so if that is missing, then return the original feed.

    This does not always give accurate results.
    The algorithm works as follows.
    Compute the ``shape_dist_traveled`` field by using Shapely to
    measure the distance of a stop along its trip LineString.
    If for a given trip this process produces a non-monotonically
    increasing, hence incorrect, list of (cumulative) distances, then
    fall back to estimating the distances as follows.

    Set the first distance to 0, the last to the length of the trip shape,
    and leave the remaining ones computed above.
    Choose the longest increasing subsequence of that new set of
    distances and use them and their corresponding departure times to linearly
    interpolate the rest of the distances.
    """
    if feed.shapes is None or feed.shapes.empty:
        return feed

    # Get stop and shape geometries as dictionaries
    geom_by_stop = feed.build_geometry_by_stop(use_utm=True)
    geom_by_shape = feed.build_geometry_by_shape(use_utm=True)

    # Memoize distance by stop by shape to avoid repeating calculations
    dist_by_stop_by_shape = {shape: {} for shape in geom_by_shape}

    def compute_dist(group):
        g = group.copy()

        # Compute the distances of the stops along this trip and memoize.
        shape = g.shape_id.iat[0]
        if pd.isna(shape):
            g["shape_dist_traveled"] = np.nan
        else:
            linestring = geom_by_shape[shape]
            dists = []
            for stop in g.stop_id.values:
                if stop in dist_by_stop_by_shape[shape]:
                    d = dist_by_stop_by_shape[shape][stop]
                else:
                    d = linestring.project(geom_by_stop[stop])
                    dist_by_stop_by_shape[shape][stop] = d
                dists.append(d)

            s = sorted(dists)
            D = linestring.length
            dists_are_reasonable = all([d < D + 100 for d in dists])

            if dists_are_reasonable and s == dists:
                # Good
                g["shape_dist_traveled"] = dists
            elif dists_are_reasonable and s == dists[::-1]:
                # Good after reversal.
                # This happens when the direction of the linestring
                # opposes the direction of the vehicle trip.
                dists = dists[::-1]
                g["shape_dist_traveled"] = dists
            else:
                # Bad. Redo using interpolation on a good subset of dists.
                dists = np.array([0] + dists[1:-1] + [D])
                ix = hp.longest_subsequence(dists, index=True)
                good_dists = np.take(dists, ix)
                g["shape_dist_traveled"] = np.interp(
                    g["departure_time_s"], g.iloc[ix]["departure_time_s"], good_dists
                )

                # Update dist dictionary with new and improved dists
                for row in g[["stop_id", "shape_dist_traveled"]].itertuples(index=False):
                    dist_by_stop_by_shape[shape][row.stop_id] = row.shape_dist_traveled

        return g

    cols = [c for c in feed.stop_times.columns if c != "shape_dist_traveled"]
    new_cols = cols + ["shape_dist_traveled"]

    m_to_dist = hp.get_convert_dist("m", feed.dist_units)
    st = (
        feed.stop_times.filter(cols)
        .merge(feed.trips.filter(["trip_id", "shape_id"]))
        # Convert departure times to seconds to ease calculatios
        .assign(departure_time_s=lambda x: x.departure_time.map(hp.timestr_to_seconds))
        .sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")
        .apply(compute_dist, include_groups=False)
        .reset_index()
        # Convert distances from meters to feed's distance units
        .assign(
            shape_dist_traveled=lambda x: x.shape_dist_traveled.map(
                m_to_dist, na_action="ignore"
            )
        )
        .filter(new_cols)
    )

    # Create new feed
    new_feed = feed.copy()
    new_feed.stop_times = st

    return new_feed


def get_start_and_end_times(feed: "Feed", date: str | None = None) -> list[str]:
    """
    Return the first departure time and last arrival time
    (HH:MM:SS time strings) listed in ``feed.stop_times``, respectively.
    Restrict to the given date (YYYYMMDD string) if specified.
    """
    st = feed.get_stop_times(date)
    return (st["departure_time"].dropna().min(), st["arrival_time"].dropna().max())


def stop_times_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str] | None = None,
) -> dict:
    """
    Return a GeoJSON FeatureCollection (in WGS84 coordinates) of Point features
    representing all the trip-stop pairs in ``feed.stop_times``.

    For every trip, drop duplicate stop IDs within that trip.
    In particular, a looping trip will lack its final stop.

    If an iterable of trip IDs is given, then subset to those trips, which could yield
    an empty FeatureCollection in case of all invalid trips.
    """
    from .stops import get_stops

    st = feed.stop_times
    if trip_ids is not None:
        st = st.loc[lambda x: x.trip_id.isin(trip_ids)]

    g = (
        get_stops(feed, as_gdf=True)
        .drop_duplicates("stop_id")
        .merge(st)
        .sort_values(["trip_id", "stop_sequence"])
        .drop_duplicates(subset=["trip_id", "stop_id"])
    )

    return json.loads(g.to_json(drop_id=True))
