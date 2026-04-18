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


def merge_feeds(  # TODO: Sketched this out, to complete.
    feed_0: Feed,
    feed_1: Feed,
    prefix_0: str = "0_",
    prefix_1: str = "1_",
    merge_similar_stops: bool = False,
    merge_similar_routes: bool = False,
    merge_similar_calendars: bool = False,
    **kwargs
) -> Tuple[Feed, Optional[Dict]]:
    """
    Merge two GTFS feeds.

    This function merges two GTFS feeds by prefixing all IDs to avoid conflicts.
    It then optionally, merges similar entities (stops, routes, calendars) based
    on provided arguments. Also returns a list of potential conflicts.

    Args:
        feed_0: The first feed to merge
        feed_1: The second feed to merge
        prefix_0: Prefix to add to all IDs in feed_0
        prefix_1: Prefix to add to all IDs in feed_1
        merge_similar_stops: Merge stops with matching IDs
        merge_similar_routes: Merge routes with same attributes
        merge_similar_calendars: Merge calendars with same service patterns
        **kwargs: Additional keyword arguments

    Returns:
        A tuple of (merged GTFS feed, conflicts dict or None)
    """

    conflicts = {}

    # Prefix all IDs in both feeds to avoid conflicts
    p_fd0 = cn.prefix_feed_ids(feed_0, prefix_0)
    p_fd1 = cn.prefix_feed_ids(feed_1, prefix_1)

    # Merge similars
    if merge_similar_stops:
        p_fd0, p_fd1, conflicts['stop_id'] = merge_similar_stops(
            p_fd0, p_fd1, prefix_0, prefix_1
        )

    if merge_similar_routes:
        pass
        # p_fd0, p_fd1, conflicts['routes'] = _merge_similar_routes(
        #     p_fd0, p_fd1
        # )

    if merge_similar_calendars:
        pass
        # p_fd0, p_fd1, conflicts['calendars'] = _merge_similar_calendars(
        #     feed_0_prefixed, feed_1_prefixed
        # )

    # TODO: add option for merging stops by distance

    # Concatenate tables
    # merged_feed = _concatenate_feeds(p_fd0, p_fd1)

    # # Clean parent_station references
    # # merged_feed = _cleanup_parent_stations(merged_feed)

    # return merged_feed, conflicts


def remap_ids(feed: Feed, id_mapping: Dict[str, str], id_type: str) -> Feed:
    """
    Replace IDs in a feed according to ``id_mapping``.
    """
    fd = feed.copy()

    for table_name, id_col in cs.ID_REMAP[id_type]:
        table = getattr(fd, table_name, None)
        if table is None:
            continue

        if id_col in table.columns:
            table[id_col] = table[id_col].map(
                lambda x: id_mapping.get(x, x)
            )

    return fd


def merge_similar_stops(
    feed_0: Feed,
    feed_1: Feed,
    prefix_0: str="",
    prefix_1: str=""
) -> tuple[Feed, Feed, list[dict]]:
    """
    Merge stops with matching IDs after removing prefixes

    This function identifies stops that have the same underlying ID
    (e.g., '0_stop_A' and '1_stop_A' both map to 'stop_A').
    """

    conflicts = []

    if feed_0.stops is None or feed_1.stops is None:
        return feed_0, feed_1, conflicts

    s0 = feed_0.stops.copy().set_index('stop_id')
    s1 = feed_1.stops.copy().set_index('stop_id')

    # Create a helper column for matching
    s0['base_id'] = [i[len(prefix_0):] if i.startswith(prefix_0) else None for i in s0.index]
    s1['base_id'] = [i[len(prefix_1):] if i.startswith(prefix_1) else None for i in s1.index]

    # Find where base_ids overlap (excluding None)
    matches = s1.reset_index().merge(
        s0.reset_index()[['stop_id', 'base_id', 'stop_name']],
        on='base_id',
        suffixes=('_1', '_0')
    ).dropna(subset=['base_id'])

    if matches.empty:
        return feed_0, feed_1, []

    # Create the mapping for _remap_ids
    stop_mapping = dict(zip(matches['stop_id_1'], matches['stop_id_0']))

    conflicts = matches[[
        'stop_id_0', 'stop_id_1', 'base_id', 'stop_name_0', 'stop_name_1'
    ]].to_dict('records')

    # Update feed_1 references
    fd1 = feed_1.remap_ids(stop_mapping, 'stop_id')

    fd1.stops = fd1.stops[~fd1.stops['stop_id'].isin(stop_mapping.keys())]

    return feed_0, fd1, conflicts
