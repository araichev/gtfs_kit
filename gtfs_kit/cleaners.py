"""
Functions about cleaning feeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip the whitespace from all column names in the given DataFrame
    and return the result.
    """
    df.columns = [col.strip() for col in df.columns]
    return df


def clean_ids(feed: "Feed") -> "Feed":
    """
    In the given Feed, strip whitespace from all string IDs and
    then replace every remaining whitespace chunk with an underscore.
    Return the resulting Feed.
    """
    # Alter feed inputs only, and build a new feed from them.
    # The derived feed attributes, such as feed.trips_i,
    # will be automatically handled when creating the new feed.
    feed = feed.copy()

    for table, d in cs.DTYPES.items():
        f = getattr(feed, table)
        if f is None:
            continue
        for column in d:
            if column in f.columns and column.endswith("_id"):
                try:
                    f[column] = (
                        f[column].str.strip().str.replace(r"\s+", "_", regex=True)
                    )
                    setattr(feed, table, f)
                except AttributeError:
                    # Column is not of string type
                    continue

    return feed


def extend_id(feed: "Feed", id_col: str, extension: str, *, prefix=True) -> "Feed":
    """
    Add a prefix (if ``prefix``) or a suffix (otherwise) to all values of column
    ``id_col`` across all tables of this Feed.
    This can be helpful when preparing to merge multiple GTFS feeds with colliding
    route IDs, say.

    Raises a ValueError if ``id_col`` values are strings,
    e.g. if ``id_col`` is 'direction_id'.
    """
    feed = feed.copy()

    for table, d in cs.DTYPES.items():
        t = getattr(feed, table)
        if t is not None and id_col in d:
            if d[id_col] != "string":
                raise ValueError(f"{id_col} must be a string column")
            else:
                t[id_col] = extension + t[id_col] if prefix else t[id_col] + extension
                setattr(feed, table, t)

    return feed


def clean_times(feed: "Feed") -> "Feed":
    """
    In the given Feed, convert H:MM:SS time strings to HH:MM:SS time
    strings to make sorting by time work as expected.
    Return the resulting Feed.
    """

    def reformat(t):
        if pd.isna(t):
            return t
        t = t.strip()
        if len(t) == 7:
            t = "0" + t
        return t

    feed = feed.copy()
    tables_and_columns = [
        ("stop_times", ["arrival_time", "departure_time"]),
        ("frequencies", ["start_time", "end_time"]),
    ]
    for table, columns in tables_and_columns:
        f = getattr(feed, table)
        if f is not None:
            f[columns] = f[columns].map(reformat)
        setattr(feed, table, f)

    return feed


def drop_zombies(feed: "Feed") -> "Feed":
    """
    In the given Feed, do the following in order and return the resulting Feed.

    1. Drop agencies with no routes.
    2. Drop stops of location type 0 or NaN with no stop times.
    3. Remove undefined parent stations from the ``parent_station`` column.
    4. Drop trips with no stop times.
    5. Drop shapes with no trips.
    6. Drop routes with no trips.
    7. Drop services with no trips.

    """
    feed = feed.copy()

    # Drop agencies with no routes
    f = feed.agency
    feed.agency = f[f["agency_id"].isin(feed.routes["agency_id"].unique())].reset_index(
        drop=True
    )

    # Drop stops of location type 0 or NaN with no stop times.
    f = feed.stops
    ids = feed.stop_times["stop_id"].unique()
    cond = f.stop_id.isin(ids)
    if "location_type" in f.columns:
        cond |= ~f.location_type.isin([0, np.nan])
    feed.stops = f[cond].reset_index(drop=True)

    # Remove undefined parent stations from the ``parent_station`` column
    if "parent_station" in feed.stops.columns:
        f = feed.stops.copy()
        ids = f["stop_id"].unique()
        f["parent_station"] = f["parent_station"].map(
            lambda x: x if x in ids else np.nan
        )
        feed.stops = f

    # Drop trips with no stop times
    f = feed.trips
    ids = feed.stop_times["trip_id"].unique()
    feed.trips = f[f["trip_id"].isin(ids)].reset_index(drop=True)

    # Drop shapes with no trips
    f = feed.shapes
    if f is not None:
        feed.shapes = f[
            f["shape_id"].isin(feed.trips["shape_id"].unique())
        ].reset_index(drop=True)

    # Drop routes with no trips
    f = feed.routes
    feed.routes = f[f["route_id"].isin(feed.trips["route_id"].unique())].reset_index(
        drop=True
    )

    # Drop services with no trips
    ids = feed.trips["service_id"].unique()
    if feed.calendar is not None:
        f = feed.calendar
        feed.calendar = f[f["service_id"].isin(ids)].reset_index(drop=True)
    if feed.calendar_dates is not None:
        f = feed.calendar_dates
        feed.calendar_dates = f[f["service_id"].isin(ids)].reset_index(drop=True)

    return feed


def clean_route_short_names(feed: "Feed") -> "Feed":
    """
    In ``feed.routes``, assign 'n/a' to missing route short names and
    strip whitespace from route short names.
    Then disambiguate each route short name that is duplicated by
    appending '-' and its route ID.
    Return the resulting Feed.
    """
    feed = feed.copy()
    r = feed.routes
    if r is None:
        return feed

    # Fill NaNs and strip whitespace
    r["route_short_name"] = r["route_short_name"].fillna("n/a").str.strip()

    # Disambiguate
    def disambiguate(row):
        rsn, rid = row
        return rsn + "-" + rid

    r["dup"] = r["route_short_name"].duplicated(keep=False)
    r.loc[r["dup"], "route_short_name"] = r.loc[
        r["dup"], ["route_short_name", "route_id"]
    ].apply(disambiguate, axis=1)
    del r["dup"]

    feed.routes = r
    return feed


def build_aggregate_routes_dict(
    routes: pd.DataFrame, by: str = "route_short_name", route_id_prefix: str = "route_"
) -> dict[str, str]:
    """
    Given a DataFrame of routes, group the routes by route short name, say,
    and assign new route IDs using the given prefix.
    Return a dictionary of the form <old route ID> -> <new route ID>.
    Helper function for :func:`aggregate_routes`.

    More specifically, group ``routes`` by the ``by`` column, and for each group make
    one new route ID for all the old route IDs in that group based on the given
    ``route_id_prefix`` string and a running count, e.g. ``'route_013'``.
    """
    if by not in routes.columns:
        raise ValueError(f"Column {by} not in routes.")

    # Create new route IDs
    n = routes.groupby(by).ngroups
    nids = hp.make_ids(n, route_id_prefix)
    nid_by_oid = dict()
    i = 0
    for col, group in routes.groupby(by):
        d = {oid: nids[i] for oid in group.route_id.values}
        nid_by_oid.update(d)
        i += 1

    return nid_by_oid


def aggregate_routes(
    feed: "Feed", by: str = "route_short_name", route_id_prefix: str = "route_"
) -> "Feed":
    """
    Aggregate routes by route short name, say, and assign new route IDs using the
    given prefix.

    More specifically, create new route IDs with the function
    :func:`build_aggregate_routes_dict` and the parameters ``by`` and
    ``route_id_prefix`` and update the old route IDs to the new ones in all the relevant
    Feed tables.
    Return the resulting Feed.
    """
    feed = feed.copy()

    # Make new route IDs
    routes = feed.routes
    nid_by_oid = build_aggregate_routes_dict(routes, by, route_id_prefix)

    # Update route IDs in routes
    routes["route_id"] = routes.route_id.map(lambda x: nid_by_oid[x])
    routes = routes.groupby(by).first().reset_index()
    feed.routes = routes

    # Update route IDs in trips
    trips = feed.trips
    trips["route_id"] = trips.route_id.map(lambda x: nid_by_oid[x])
    feed.trips = trips

    # Update route IDs of fare rules
    if feed.fare_rules is not None and "route_id" in feed.fare_rules.columns:
        fr = feed.fare_rules
        fr["route_id"] = fr.route_id.map(lambda x: nid_by_oid[x])
        feed.fare_rules = fr

    return feed


def build_aggregate_stops_dict(
    stops: pd.DataFrame, by: str = "stop_code", stop_id_prefix: str = "stop_"
) -> dict[str, str]:
    """
    Given a DataFrame of stops, group the stops by stop code, say,
    and assign new stop IDs using the given prefix.
    Return a dictionary of the form <old stop ID> -> <new stop ID>.
    Helper function for :func:`aggregate_stops`.

    More specifically, group ``stops`` by the ``by`` column, and for each group make
    one new stop ID for all the old stops IDs in that group based on the given
    ``stop_id_prefix`` string and a running count, e.g. ``'stop_013'``.
    """
    if by not in stops.columns:
        raise ValueError(f"Column {by} not in stops.")

    # Create new stop IDs
    n = stops.groupby(by).ngroups
    nids = hp.make_ids(n, stop_id_prefix)
    nid_by_oid = dict()
    i = 0
    for col, group in stops.groupby(by):
        d = {oid: nids[i] for oid in group.stop_id.values}
        nid_by_oid.update(d)
        i += 1

    return nid_by_oid


def aggregate_stops(
    feed: "Feed", by: str = "stop_code", stop_id_prefix: str = "stop_"
) -> "Feed":
    """
    Aggregate stops by stop code, say, and assign new stop IDs using the
    given prefix.

    More specifically, create new stop IDs with the function
    :func:`build_aggregate_stops_dict` and the parameters ``by`` and
    ``stop_id_prefix`` and update the old stop IDs to the new ones in all the relevant
    Feed tables.
    Return the resulting Feed.
    """
    feed = feed.copy()

    # Make new stop ID by old stop ID dict
    stops = feed.stops
    nid_by_oid = build_aggregate_stops_dict(stops, by, stop_id_prefix)

    # Apply dict
    stops["stop_id"] = stops.stop_id.map(nid_by_oid)
    if "parent_station" in stops:
        stops["parent_station"] = stops.parent_station.map(nid_by_oid)

    stops = stops.groupby(by).first().reset_index()
    feed.stops = stops

    # Update stop IDs of stop times
    stop_times = feed.stop_times
    stop_times["stop_id"] = stop_times.stop_id.map(lambda x: nid_by_oid[x])
    feed.stop_times = stop_times

    # Update route IDs of transfers
    if feed.transfers is not None:
        transfers = feed.transfers
        transfers["to_stop_id"] = transfers.to_stop_id.map(lambda x: nid_by_oid[x])
        transfers["from_stop_id"] = transfers.from_stop_id.map(lambda x: nid_by_oid[x])
        feed.transfers = transfers

    return feed


def clean(feed: "Feed") -> "Feed":
    """
    Apply the following functions to the given Feed in order and return the resulting
    Feed.

    #. :func:`clean_ids`
    #. :func:`clean_times`
    #. :func:`clean_route_short_names`
    #. :func:`drop_zombies`

    """
    feed = feed.copy()
    ops = ["clean_ids", "clean_times", "clean_route_short_names", "drop_zombies"]
    for op in ops:
        feed = globals()[op](feed)

    return feed


def drop_invalid_columns(feed: "Feed") -> "Feed":
    """
    Drop all DataFrame columns of the given Feed that are not
    listed in the GTFS.
    Return the resulting Feed.
    """
    feed = feed.copy()
    for table, d in cs.DTYPES.items():
        f = getattr(feed, table)
        if f is None:
            continue
        valid_columns = set(d.keys())
        for col in f.columns:
            if col not in valid_columns:
                print(f"{table}: dropping invalid column {col}")
                del f[col]
        setattr(feed, table, f)

    return feed
