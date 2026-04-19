"""
Functions to allow merging feeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg

from . import cleaners as cn
from . import constants as cs

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def merge_feeds(
    feed_0: Feed,
    feed_1: Feed,
    prefix_0: str = "0_",
    prefix_1: str = "1_",
    merge_stop_ids: bool = True,
    merge_route_names: bool = True,
    merge_calendars: bool = True,
    **kwargs,
) -> tuple[Feed, dict]:
    """
    Merge two GTFS feeds.

    This function merges two GTFS feeds by prefixing all ID fields to avoid
    conflicts. It then optionally merges similar entities (stops, routes,
    calendars) based on provided arguments. Also returns a list of potential
    conflicting entities.

    Args:
        feed_0: The first feed to merge
        feed_1: The second feed to merge
        prefix_0: Prefix to add to all IDs in feed_0
        prefix_1: Prefix to add to all IDs in feed_1
        merge_stops_ids: Merge stops with matching IDs
        merge_route_names: Merge routes with similar attributes
            (route_short_name filled with route_long_name, route_type)
        merge_calendars: Merge calendars with same service patterns
            (same days active, same start and end dates)
        **kwargs: Additional keyword arguments for any of the sub-functions

    Returns:
        A tuple of merged GTFS feed and a conflicts dict
    """

    conflicts = {}

    # Prefix all IDs in both feeds to avoid conflicts
    p_fd0 = cn.prefix_feed_ids(feed_0.copy(), prefix_0)
    p_fd1 = cn.prefix_feed_ids(feed_1.copy(), prefix_1)

    # Merge similars
    if merge_stop_ids:
        p_fd0, p_fd1, conflicts["stop_id"] = merge_similar_stops(
            p_fd0, p_fd1, prefix_0, prefix_1, **kwargs
        )

    if merge_route_names:
        p_fd0, p_fd1, conflicts["routes"] = merge_similar_routes(
            p_fd0, p_fd1, **kwargs
        )

    if merge_calendars:
        p_fd0, p_fd1, conflicts["calendars"] = merge_similar_calendars(
            p_fd0, p_fd1, **kwargs
        )

    # TODO: add option for merging stops by distance

    # Concatenate tables
    merged_feed = concatenate_feeds(p_fd0, p_fd1)

    # Clean parent_station references
    merged_feed = cn.clean_parent_stations(merged_feed)

    return merged_feed, conflicts


def remap_ids(feed: Feed, id_mapping: dict[str, str], id_type: str) -> Feed:
    """
    Replace IDs in a feed according to ``id_mapping``.
    """
    fd = feed.copy()

    for table_name, id_col in cs.ID_REMAP[id_type]:
        table = getattr(fd, table_name, None)
        if table is None:
            continue

        if id_col in table.columns:
            table[id_col] = table[id_col].map(lambda x: id_mapping.get(x, x))

    return fd


def merge_similar_stops(
    feed_0: Feed, feed_1: Feed, prefix_0: str = "", prefix_1: str = "", **kwargs
) -> tuple[Feed, Feed, list[dict]]:
    """
    Merge stops with matching IDs after removing prefixes

    This function identifies stops that have the same underlying ID
    (e.g., '0_stop_A' and '1_stop_A' both map to 'stop_A').
    """

    conflicts = []

    s0 = feed_0.stops.copy().set_index("stop_id")
    s1 = feed_1.stops.copy().set_index("stop_id")

    # create a helper column for matching
    s0["match_key"] = [
        i[len(prefix_0) :] if i.startswith(prefix_0) else None for i in s0.index
    ]
    s1["match_key"] = [
        i[len(prefix_1) :] if i.startswith(prefix_1) else None for i in s1.index
    ]

    # find where base_ids overlap (excluding None)
    matches = (
        s1.reset_index()
        .merge(
            s0.reset_index()[["stop_id", "match_key", "stop_name"]],
            on="match_key",
            suffixes=("_1", "_0"),
        )
        .dropna(subset=["match_key"])
    )

    if matches.empty:
        return feed_0, feed_1, []

    # create the mapping for remap_ids
    stop_mapping = dict(zip(matches["stop_id_1"], matches["stop_id_0"]))

    conflicts = matches[
        ["stop_id_0", "stop_id_1", "match_key", "stop_name_0", "stop_name_1"]
    ].to_dict("records")

    # remap and cleanup
    fd1 = feed_1.copy()

    ids_to_remove = set(stop_mapping.keys())
    fd1.stops = fd1.stops[~fd1.stops["stop_id"].isin(ids_to_remove)]
    fd1 = fd1.remap_ids(stop_mapping, "stop_id")

    return feed_0, fd1, conflicts


def merge_similar_routes(
    feed_0: Feed, feed_1: Feed, **kwargs
) -> tuple[Feed, Feed, list[dict]]:
    """
    Merge routes based on Agency, Short Name (or Long Name), and Type.
    """
    conflicts = []

    r0 = feed_0.routes.copy()
    r1 = feed_1.routes.copy()

    def match_key(df):  # helper column for matching
        name_col = df["route_short_name"].fillna(df["route_long_name"])
        return name_col.astype(str) + "_" + df["route_type"].astype(str)

    r0["match_key"] = match_key(r0)
    r1["match_key"] = match_key(r1)

    matches = r1.merge(
        r0[["route_id", "match_key", "route_short_name", "route_type"]],
        on="match_key",
        suffixes=("_1", "_0"),
    )

    if matches.empty:
        return feed_0, feed_1, []

    # create mapping and conflict report
    route_mapping = dict(zip(matches["route_id_1"], matches["route_id_0"]))

    conflicts = matches[
        ["route_id_0", "route_id_1", "route_short_name_0", "route_type_0"]
    ].to_dict("records")

    # remap and cleanup
    fd1 = feed_1.copy()

    ids_to_remove = set(route_mapping.keys())
    fd1.routes = fd1.routes[~fd1.routes["route_id"].isin(ids_to_remove)]
    fd1 = fd1.remap_ids(route_mapping, "route_id")

    return feed_0, fd1, conflicts


def merge_similar_calendars(
    feed_0: Feed, feed_1: Feed, ignore_dates: bool = False, **kwargs
) -> tuple[Feed, Feed, list[dict]]:
    """
    Merges similar calendars based on their day schedule.
    If ``ignore_dates``, ignore calendar start and end dates.
    """
    conflicts = []

    c0 = feed_0.calendar.copy()
    c1 = feed_1.calendar.copy()

    match_cols = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    if not ignore_dates:
        match_cols += ["start_date", "end_date"]

    matches = c1.merge(c0, on=match_cols, suffixes=("_1", "_0"))

    if matches.empty:
        return feed_0, feed_1, []

    # create mapping and conflict report
    service_mapping = dict(zip(matches["service_id_1"], matches["service_id_0"]))
    conflicts = matches[["service_id_0", "service_id_1"] + match_cols].to_dict("records")

    # remap and cleanup
    fd1 = feed_1.copy()

    ids_to_remove = set(service_mapping.keys())
    fd1.calendar = fd1.calendar[~fd1.calendar["service_id"].isin(ids_to_remove)]
    fd1 = fd1.remap_ids(service_mapping, "service_id")

    return feed_0, fd1, conflicts


def concatenate_feeds(feed_0: Feed, feed_1: Feed) -> Feed:
    """
    Concatenate two feeds by combining all their tables. This ignores conflicts.
    """
    fd = feed_0.copy()

    # merge each table
    for table_name in cs.DTYPES:
        df_0 = getattr(feed_0, table_name, None)
        df_1 = getattr(feed_1, table_name, None)

        if df_0 is not None and df_1 is not None:
            setattr(fd, table_name, pd.concat([df_0, df_1], ignore_index=True))
        elif df_0 is not None:
            setattr(fd, table_name, df_0.copy())
        elif df_1 is not None:
            setattr(fd, table_name, df_1.copy())
        else:
            setattr(fd, table_name, None)

    return fd
