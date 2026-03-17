"""
Functions about miscellany.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def list_fields(feed: "Feed", table: str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame describing all the fields of the GTFS tables in the given feed
    or in the given table if specified.

    The resulting DataFrame has the following columns.

    - ``'table'``: name of the GTFS table, e.g. ``'stops'``
    - ``'column'``: name of a column in the table,
      e.g. ``'stop_id'``
    - ``'num_values'``: number of values in the column
    - ``'num_nonnull_values'``: number of nonnull values in the
      column
    - ``'num_unique_values'``: number of unique values in the
      column, excluding null values
    - ``'min_value'``: minimum value in the column
    - ``'max_value'``: maximum value in the column

    If the table is not in the feed, then return an empty DataFrame
    If the table is not valid, raise a ValueError
    """
    gtfs_tables = list(cs.DTYPES.keys())
    if table is not None:
        if table not in gtfs_tables:
            raise ValueError(f"{table} is not a GTFS table")
        else:
            tables = [table]
    else:
        tables = gtfs_tables

    frames = []
    for table in tables:
        f = getattr(feed, table)
        if f is None:
            continue

        def my_agg(col):
            d = {}
            d["column"] = col.name
            d["num_values"] = col.size
            d["num_nonnull_values"] = col.count()
            d["num_unique_values"] = col.nunique()
            d["min_value"] = col.dropna().min()
            d["max_value"] = col.dropna().max()
            return pd.Series(d)

        g = f.apply(my_agg).T.reset_index(drop=True)
        g["table"] = table
        frames.append(g)

    cols = [
        "table",
        "column",
        "num_values",
        "num_nonnull_values",
        "num_unique_values",
        "min_value",
        "max_value",
    ]

    if not frames:
        f = pd.DataFrame()
    else:
        f = pd.concat(frames)
        # Rearrange columns
        f = f[cols].copy()

    return f


def describe(feed: "Feed", sample_date: str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of routes.
    Specialize some those indicators to the given YYYYMMDD sample date string,
    e.g. number of routes active on the date.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    """
    from . import calendar as cl

    d = dict()
    dates = cl.get_dates(feed)
    d["agencies"] = feed.agency["agency_name"].tolist()
    d["timezone"] = feed.agency["agency_timezone"].iat[0]
    d["start_date"] = dates[0]
    d["end_date"] = dates[-1]
    d["num_routes"] = feed.routes.shape[0]
    d["num_trips"] = feed.trips.shape[0]
    d["num_stops"] = feed.stops.shape[0]
    if feed.shapes is not None:
        d["num_shapes"] = feed.shapes["shape_id"].nunique()
    else:
        d["num_shapes"] = 0

    if sample_date is None or sample_date not in feed.get_dates():
        sample_date = cl.get_first_week(feed)[3]
    d["sample_date"] = sample_date
    d["num_routes_active_on_sample_date"] = feed.get_routes(sample_date).shape[0]
    trips = feed.get_trips(sample_date)
    d["num_trips_active_on_sample_date"] = trips.shape[0]
    d["num_stops_active_on_sample_date"] = feed.get_stops(sample_date).shape[0]
    f = pd.DataFrame(list(d.items()), columns=["indicator", "value"])

    return f


def assess_quality(feed: "Feed") -> pd.DataFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of trips missing shapes.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    This function is odd but useful for seeing roughly how broken a feed is
    This function is not a GTFS validator.
    """
    d = dict()

    # Count duplicate route short names
    r = feed.routes
    dup = r.duplicated(subset=["route_short_name"])
    n = dup[dup].count()
    d["num_route_short_names_duplicated"] = n
    d["frac_route_short_names_duplicated"] = n / r.shape[0]

    # Count stop times missing shape_dist_traveled values
    st = feed.stop_times.sort_values(["trip_id", "stop_sequence"])
    if "shape_dist_traveled" in st.columns:
        # Count missing distances
        n = st[st["shape_dist_traveled"].isnull()].shape[0]
        d["num_stop_time_dists_missing"] = n
        d["frac_stop_time_dists_missing"] = n / st.shape[0]
    else:
        d["num_stop_time_dists_missing"] = st.shape[0]
        d["frac_stop_time_dists_missing"] = 1

    # Count direction_ids missing
    t = feed.trips
    if "direction_id" in t.columns:
        n = t[t["direction_id"].isnull()].shape[0]
        d["num_direction_ids_missing"] = n
        d["frac_direction_ids_missing"] = n / t.shape[0]
    else:
        d["num_direction_ids_missing"] = t.shape[0]
        d["frac_direction_ids_missing"] = 1

    # Count trips missing shapes
    if feed.shapes is not None:
        n = t[t["shape_id"].isnull()].shape[0]
    else:
        n = t.shape[0]
    d["num_trips_missing_shapes"] = n
    d["frac_trips_missing_shapes"] = n / t.shape[0]

    # Count missing departure times
    n = st[st["departure_time"].isnull()].shape[0]
    d["num_departure_times_missing"] = n
    d["frac_departure_times_missing"] = n / st.shape[0]

    # Count missing first departure times missing
    g = st.groupby("trip_id").first().reset_index()
    n = g[g["departure_time"].isnull()].shape[0]
    d["num_first_departure_times_missing"] = n
    d["frac_first_departure_times_missing"] = n / st.shape[0]

    # Count missing last departure times
    g = st.groupby("trip_id").last().reset_index()
    n = g[g["departure_time"].isnull()].shape[0]
    d["num_last_departure_times_missing"] = n
    d["frac_last_departure_times_missing"] = n / st.shape[0]

    # Opine
    if (
        (d["frac_first_departure_times_missing"] >= 0.1)
        or (d["frac_last_departure_times_missing"] >= 0.1)
        or d["frac_trips_missing_shapes"] >= 0.8
    ):
        d["assessment"] = "bad feed"
    elif (
        d["frac_direction_ids_missing"]
        or d["frac_stop_time_dists_missing"]
        or d["num_route_short_names_duplicated"]
    ):
        d["assessment"] = "probably a fixable feed"
    else:
        d["assessment"] = "good feed"

    f = pd.DataFrame(list(d.items()), columns=["indicator", "value"])

    return f


def convert_dist(feed: "Feed", new_dist_units: str) -> "Feed":
    """
    Convert the distances recorded in the ``shape_dist_traveled``
    columns of the given Feed to the given distance units.
    New distance units must lie in :const:`.constants.DIST_UNITS`.
    Return the resulting Feed.
    """
    feed = feed.copy()

    if feed.dist_units == new_dist_units:
        # Nothing to do
        return feed

    old_dist_units = feed.dist_units
    feed.dist_units = new_dist_units

    converter = hp.get_convert_dist(old_dist_units, new_dist_units)

    if hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        feed.stop_times["shape_dist_traveled"] = feed.stop_times[
            "shape_dist_traveled"
        ].map(converter)

    if hp.is_not_null(feed.shapes, "shape_dist_traveled"):
        feed.shapes["shape_dist_traveled"] = feed.shapes["shape_dist_traveled"].map(
            converter
        )

    return feed


def compute_network_stats_0(
    stop_times_subset: pd.DataFrame,
    trip_stats_subset: pd.DataFrame,
    *,
    split_route_types=False,
) -> pd.DataFrame:
    """
    Compute some network stats for the trips common to the
    given subset of stop times and given subses of trip stats
    of the form output by the function :func:`.trips.compute_trip_stats`

    Return a DataFrame with the columns

    - ``'route_type'`` (optional): presest if and only if ``split_route_types``
    - ``'num_stops'``: number of stops active on the date
    - ``'num_routes'``: number of routes active on the date
    - ``'num_trips'``: number of trips that start on the date
    - ``'num_trip_starts'``: number of trips with nonnull start
      times on the date
    - ``'num_trip_ends'``: number of trips with nonnull start times
      and nonnull end times on the date, ignoring trips that end
      after 23:59:59 on the date
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service on the date
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs on the date
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs on the date
    - ``'service_distance'``: sum of the service distances for the
      active routes on the date;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_duration'``: sum of the service durations for the
      active routes on the date; measured in hours
    - ``'service_speed'``: service_distance/service_duration on the
      date

    Exclude dates with no active stops, which could yield the empty DataFrame.

    Helper function for :func:`compute_network_stats`.
    """
    final_cols = [
        "num_stops",
        "num_routes",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "peak_num_trips",
        "peak_start_time",
        "peak_end_time",
        "service_distance",
        "service_duration",
        "service_speed",
    ]
    if split_route_types:
        final_cols.insert(0, "route_type")

    null_stats = pd.DataFrame(columns=final_cols)

    # Handle defunct case
    if stop_times_subset.empty or trip_stats_subset.empty:
        return null_stats

    # Handle generic case
    trip_ids = set(trip_stats_subset["trip_id"].values) & set(
        stop_times_subset["trip_id"].values
    )
    ts = trip_stats_subset.loc[lambda x: x["trip_id"].isin(trip_ids)].copy()
    st = stop_times_subset.loc[lambda x: x["trip_id"].isin(trip_ids)].copy()

    # Convert timestrings to seconds for quicker calculations
    ts[["start_time", "end_time"]] = ts[["start_time", "end_time"]].map(
        hp.timestr_to_seconds
    )

    def agg(g: pd.DataFrame, route_type: str | None = None) -> dict:
        d = {}
        if route_type is not None:
            d["route_type"] = route_type
        d["num_stops"] = st.loc[
            lambda x: x["trip_id"].isin(g["trip_id"]), "stop_id"
        ].nunique()
        d["num_routes"] = g["route_id"].nunique()
        d["num_trips"] = len(g)
        d["num_trip_starts"] = g["start_time"].count()
        d["num_trip_ends"] = g.loc[g["end_time"] < 24 * 3600, "end_time"].count()
        d["service_distance"] = g["distance"].sum()
        d["service_duration"] = g["duration"].sum()
        if d["service_distance"]:
            d["service_speed"] = d["service_distance"] / d["service_duration"]
        else:
            d["service_speed"] = 0

        # Compute peak stats, which is the slowest part
        active_trips = hp.get_active_trips_df(g[["start_time", "end_time"]])
        times, counts = (active_trips.index.values, active_trips.values)
        start, end = hp.get_peak_indices(times, counts)
        d["peak_num_trips"] = counts[start]
        d["peak_start_time"] = times[start]
        d["peak_end_time"] = times[end]
        return d

    # Compute stats
    if split_route_types:
        series = []
        for route_type, g in ts.groupby("route_type"):
            series.append(pd.Series(agg(g, route_type)))

        stats = pd.DataFrame(series)

    else:
        stats = pd.DataFrame(agg(ts), index=[0])

    # Convert seconds back to timestrings
    times = ["peak_start_time", "peak_end_time"]
    stats[times] = stats[times].map(lambda t: hp.seconds_to_timestr(t))

    return stats.filter(final_cols)


def compute_network_stats(
    feed: "Feed",
    dates: list[str],
    trip_stats: pd.DataFrame | None = None,
    *,
    split_route_types=False,
) -> pd.DataFrame:
    """
    Compute some network stats for the given subset of trip stats, which defaults to
    `feed.compute_trip_stats()`, and for the given dates (YYYYMMDD date stings).

    Return a table with the columns

    - ``'date'``
    - ``'route_type'`` (optional): presest if and only if ``split_route_types``
    - ``'num_stops'``: number of stops active on the date
    - ``'num_routes'``: number of routes active on the date
    - ``'num_trips'``: number of trips that start on the date
    - ``'num_trip_starts'``: number of trips with nonnull start
      times on the date
    - ``'num_trip_ends'``: number of trips with nonnull start times
      and nonnull end times on the date, ignoring trips that end
      after 23:59:59 on the date
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service on the date
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs on the date
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs on the date
    - ``'service_distance'``: sum of the service distances for the
      active routes on the date;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_duration'``: sum of the service durations for the
      active routes on the date; measured in hours
    - ``'service_speed'``: service_distance/service_duration on the
      date

    Exclude dates with no active stops, which could yield the empty DataFrame.

    The route and trip stats for date d contain stats for trips that
    start on date d only and ignore trips that start on date d-1 and
    end on date d.

    Notes
    -----
    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.

    """
    dates = feed.subset_dates(dates)

    # Handle defunct case
    null_stats = compute_network_stats_0(
        pd.DataFrame(), pd.DataFrame(), split_route_types=split_route_types
    )
    if not dates:
        return null_stats

    final_cols = ["date"] + null_stats.columns.to_list()

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFrame.
    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()

    activity = feed.compute_trip_activity(dates)
    stats_by_ids = {}
    frames = []
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            stats = (
                stats_by_ids[ids]
                # Assign date
                .assign(date=date)
            )
        elif ids:
            # Compute stats
            ts = trip_stats.loc[lambda x: x.trip_id.isin(ids)].copy()
            stats = (
                compute_network_stats_0(
                    feed.stop_times, ts, split_route_types=split_route_types
                )
                # Assign date
                .assign(date=date)
            )
            # Memoize stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats and order columns
    return pd.concat(frames, ignore_index=True).filter(final_cols)


def compute_network_time_series(
    feed: "Feed",
    dates: list[str],
    trip_stats: pd.DataFrame | None = None,
    freq: str = "h",
    *,
    split_route_types: bool = False,
) -> pd.DataFrame:
    """
    Compute some network stats in time series form for the given dates
    (YYYYMMDD date strings) and trip stats, which defaults to
    ``feed.compute_trip_stats()``.
    Use the given Pandas frequency string ``freq`` to specify the frequency of the
    resulting time series, e.g. '5Min'.
    If ``split_route_types``, then split stats by route type; otherwise don't.

    Return a long-form time series table with the columns

    - ``'datetime'``: datetime object
    - ``'route_type'``: integer; present if and only if ``split_route_types``
    - ``'num_trips'``: number of trips in service during during the
      time period
    - ``'num_trip_starts'``: number of trips with starting during the
      time period
    - ``'num_trip_ends'``: number of trips ending during the
      time period, ignoring the trips the end past midnight
    - ``'service_distance'``: distance traveled during the time
      period by all trips active during the time period;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_duration'``: duration traveled during the time
      period by all trips active during the time period;
      measured in hours
    - ``'service_speed'``: ``service_distance/service_duration`` when defined; 0
      otherwise

    Exclude dates that lie outside of the Feed's date range.
    If all the dates given lie outside of the Feed's date range,
    then return an empty DataFrame with the specified columns.

    Notes
    -----
    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.

    """
    final_cols = [
        "datetime",
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_distance",
        "service_duration",
        "service_speed",
    ]
    if split_route_types:
        final_cols.insert(1, "route_type")
    null_stats = pd.DataFrame(columns=final_cols)

    rts = feed.compute_route_time_series(dates, trip_stats, freq=freq)

    # Handle defunct case
    if rts.empty:
        return null_stats

    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()

    if split_route_types:
        group_keys = ["datetime", "route_type"]
        rts = rts.merge(trip_stats.filter(["route_id", "route_type"]), how="left")
    else:
        group_keys = ["datetime"]

    indicators = set(final_cols) - {"datetime", "route_type"}
    f = (
        rts.groupby(group_keys)
        .agg(
            **{ind: (ind, lambda x: x.sum(min_count=1)) for ind in indicators}
        )  # All-NaNs should sum to NaN
        .reset_index()
    )
    # Recalculate service speed
    f["service_speed"] = (
        f["service_distance"]
        .div(f["service_duration"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return f


def create_shapes(feed: "Feed", *, all_trips: bool = False) -> "Feed":
    """
    Given a feed, create a shape for every trip that is missing a
    shape ID.
    Do this by connecting the stops on the trip with straight lines.
    Return the resulting feed which has updated shapes and trips
    tables.

    If ``all_trips``, then create new shapes for all trips by
    connecting stops, and remove the old shapes.
    """
    feed = feed.copy()

    if all_trips:
        trip_ids = feed.trips["trip_id"]
    else:
        trip_ids = feed.trips[feed.trips["shape_id"].isnull()]["trip_id"]

    # Get stop times for given trips
    f = feed.stop_times[feed.stop_times["trip_id"].isin(trip_ids)][
        ["trip_id", "stop_sequence", "stop_id"]
    ]
    f = f.sort_values(["trip_id", "stop_sequence"])

    if f.empty:
        # Nothing to do
        return feed

    # Create new shape IDs for given trips.
    # To do this, collect unique stop sequences,
    # sort them to impose a canonical order, and
    # assign shape IDs to them
    stop_seqs = sorted(
        set(tuple(group["stop_id"].values) for trip, group in f.groupby("trip_id"))
    )
    ids = hp.make_ids(len(stop_seqs), "shape_")
    shape_by_stop_seq = {seq: ids[i] for i, seq in enumerate(stop_seqs)}

    # Assign these new shape IDs to given trips
    shape_by_trip = {
        trip: shape_by_stop_seq[tuple(group["stop_id"].values)]
        for trip, group in f.groupby("trip_id")
    }
    trip_cond = feed.trips["trip_id"].isin(trip_ids)
    feed.trips.loc[trip_cond, "shape_id"] = feed.trips.loc[trip_cond, "trip_id"].map(
        lambda x: shape_by_trip[x]
    )

    # Build new shapes for given trips
    G = [
        [shape, i, stop]
        for stop_seq, shape in shape_by_stop_seq.items()
        for i, stop in enumerate(stop_seq)
    ]
    g = pd.DataFrame(G, columns=["shape_id", "shape_pt_sequence", "stop_id"])
    g = g.merge(feed.stops[["stop_id", "stop_lon", "stop_lat"]]).sort_values(
        ["shape_id", "shape_pt_sequence"]
    )
    g = g.drop(["stop_id"], axis=1)
    g = g.rename(columns={"stop_lon": "shape_pt_lon", "stop_lat": "shape_pt_lat"})

    if feed.shapes is not None and not all_trips:
        # Update feed shapes with new shapes
        feed.shapes = pd.concat([feed.shapes, g], sort=False)
    else:
        # Create all new shapes
        feed.shapes = g

    return feed


def compute_bounds(feed: "Feed", stop_ids: list[str] | None = None) -> np.array:
    """
    Return the bounding box (Numpy array [min longitude, min latitude, max longitude,
    max latitude]) of the given Feed's stops or of the subset of stops
    specified by the given stop IDs.
    """
    from .stops import get_stops

    g = get_stops(feed, as_gdf=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return g.total_bounds


def compute_convex_hull(feed: "Feed", stop_ids: list[str] | None = None) -> sg.Polygon:
    """
    Return a convex hull (Shapely Polygon) representing the convex hull of the given
    Feed's stops or of the subset of stops specified by the given stop IDs.
    """
    from .stops import get_stops

    g = get_stops(feed, as_gdf=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return g.union_all().convex_hull


def compute_centroid(feed: "Feed", stop_ids: list[str] | None = None) -> sg.Point:
    """
    Return the centroid (Shapely Point) of the convex hull the given Feed's stops
    or of the subset of stops specified by the given stop IDs.
    """
    from .stops import get_stops

    g = get_stops(feed, as_gdf=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return g.union_all().convex_hull.centroid


def restrict_to_trips(feed: "Feed", trip_ids: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one to only the stops,
    trips, shapes, etc. used by the trips of the given IDs.
    Return the resulting feed.

    If no valid trip IDs are given, which includes the case of the empty list,
    then the resulting feed will have all empty non-agency tables.

    This function is probably more useful internally than externally.
    """
    feed = feed.copy()
    has_agency_ids = "agency_id" in feed.routes.columns

    # Subset trips
    feed.trips = feed.trips.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Subset routes
    feed.routes = feed.routes.loc[lambda x: x.route_id.isin(feed.trips.route_id)].copy()

    # Subset stop times
    feed.stop_times = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Subset stops, collecting parent stations too
    stop_ids_0 = set(feed.stop_times["stop_id"])
    if "parent_station" in feed.stops:
        stop_ids_1 = set(
            feed.stops.loc[
                lambda x: x["stop_id"].isin(stop_ids_0), "parent_station"
            ].dropna()
        )
        stop_ids = stop_ids_0 | stop_ids_1
    else:
        stop_ids = stop_ids_0

    feed.stops = feed.stops.loc[lambda x: x["stop_id"].isin(stop_ids)].copy()

    # Subset calendar
    service_ids = feed.trips["service_id"].unique()
    if feed.calendar is not None:
        feed.calendar = feed.calendar.loc[lambda x: x.service_id.isin(service_ids)].copy()

    # Subset agency
    if has_agency_ids:
        agency_ids = feed.routes["agency_id"]
        feed.agency = feed.agency.loc[lambda x: x.agency_id.isin(agency_ids)].copy()

    # Now for the optional files.
    # Subset calendar dates.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Subset frequencies
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Subset shapes
    if feed.shapes is not None:
        shape_ids = feed.trips.shape_id
        feed.shapes = feed.shapes.loc[lambda x: x.shape_id.isin(shape_ids)].copy()

    # Subset transfers
    if feed.transfers is not None:
        feed.transfers = feed.transfers.loc[
            lambda x: x.from_stop_id.isin(stop_ids) & x.to_stop_id.isin(stop_ids)
        ].copy()

    return feed


def restrict_to_routes(feed: "Feed", route_ids: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_trips` and
    the trips with the given route IDs.
    Return the resulting feed.
    """
    trip_ids = feed.trips.loc[lambda x: x.route_id.isin(route_ids), "trip_id"].tolist()
    return restrict_to_trips(feed, trip_ids)


def restrict_to_agencies(feed: "Feed", agency_ids: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_routes` and
    the routes with the given agency IDs.
    Return the resulting feed.
    """
    # Build feed via `restrict_to_routes`
    feed = feed.copy()
    route_ids = feed.routes.loc[
        lambda x: x.agency_id.isin(agency_ids), "route_id"
    ].tolist()

    return feed.restrict_to_routes(route_ids)


def restrict_to_dates(feed: "Feed", dates: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_trips` and
    the trips active on at least one of the given dates (YYYYMMDD strings).
    Return the resulting feed.
    """
    # Get every trip that is active on at least one of the dates
    trip_activity = feed.compute_trip_activity(dates)
    if trip_activity.empty:
        trip_ids = []
    else:
        trip_ids = trip_activity.loc[
            lambda x: x.filter(dates).sum(axis=1) > 0,
            "trip_id",
        ]

    return restrict_to_trips(feed, trip_ids)


def restrict_to_area(feed: "Feed", area: gpd.GeoDataFrame) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_trips`
    and the trips that have at least one stop intersecting the given GeoDataFrame of
    polygons.
    Return the resulting feed.
    """
    from .stops import get_stops_in_area

    # Get IDs of stops within the polygon
    stop_ids = get_stops_in_area(feed, area).stop_id

    # Get all trips with at least one of those stops
    st = feed.stop_times.copy()
    trip_ids = st.loc[lambda x: x.stop_id.isin(stop_ids), "trip_id"]

    return restrict_to_trips(feed, trip_ids)


def _reshape_stop_times(stop_times: pd.DataFrame) -> pd.DataFrame:
    """
    Given a GTFS stop times DataFrame, reshape it to have only the following columns.

    - trip_id
    - stop_sequence
    - from_departure_time
    - to_departure_time
    - from_stop_id
    - to_stop_id
    - from_shape_dist_traveled (optional): present if and only if
      'shape_dist_traveled' column present in given stop times
    - to_shape_dist_traveled (optional): present if and only if 'shape_dist_traveled'
      column present in given stop times

    This is a helper function for :func:`compute_screen_line_counts`.
    """
    f = stop_times.sort_values(["trip_id", "stop_sequence"], ignore_index=True)
    g = f.groupby("trip_id")

    has_dist = "shape_dist_traveled" in f.columns

    # For each trip, create shifted columns for the "to" stop and its associated time and distance fields
    f["to_stop_id"] = g["stop_id"].shift(-1)
    f["to_departure_time"] = g["departure_time"].shift(-1)
    if has_dist:
        f["to_shape_dist_traveled"] = g["shape_dist_traveled"].shift(-1)

    # Drop rows where there is no "to" stop (i.e. the last stop in each trip)
    f = f.dropna(subset=["to_stop_id"])

    # Rename the original columns to reflect they represent the "from" stop in the segment
    f = f.rename(
        columns={
            "stop_id": "from_stop_id",
            "departure_time": "from_departure_time",
            "shape_dist_traveled": "from_shape_dist_traveled",
        }
    )

    return f.filter(
        [
            "trip_id",
            "stop_sequence",
            "from_departure_time",
            "to_departure_time",
            "from_stop_id",
            "to_stop_id",
            "from_shape_dist_traveled",
            "to_shape_dist_traveled",
        ]
    )


def compute_screen_line_counts(
    feed: "Feed",
    screen_lines: gpd.GeoDataFrame,
    dates: list[str],
    *,
    include_testing_cols: bool = False,
) -> pd.DataFrame:
    """
    Find all the Feed trips active on the given YYYYMMDD dates that intersect
    the given segment-associated screen lines of the form output by
    :func:`build_screen_lines`.
    Behind the scenes, use simple sub-LineStrings of the feed
    to compute screen line intersections.
    Using them instead of the Feed shapes avoids miscounting intersections in the
    case of non-simple (self-intersecting) shapes.

    For each trip crossing a screen line,
    compute the crossing time, crossing direction, etc. and return a DataFrame
    of results with the columns

    - ``'date'``: the YYYYMMDD date string given
    - ``'screen_line_id'``: ID of a screen line
    - ``'trip_id'``: ID of a trip that crosses the screen line
    - ``'shape_id'``: ID of the trip's shape
    - ``'direction_id'``: GTFS direction of trip
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'shape_id'``
    - ``'crossing_direction'``: 1 or -1; 1 indicates trip travel from the
      left side to the right side of the screen line;
      -1 indicates trip travel in the  opposite direction
    - ``'crossing_time'``: time, according to the GTFS schedule, that the trip
      crosses the screen line
    - ``'crossing_dist_m'``: distance along the trip shape (not subshape) of the
      crossing; in meters

    If ``include_testing_columns``, then include the following extra columns for testing
    purposes.

    - ``'subshape_id'``: ID of the simple sub-LineString S of the trip's shape that
      crosses the screen line
    - ``'subshape_length_m'``: length of S in meters
    - ``'from_departure_time'``: departure time of the trip from the last stop before
      the screen line
    - ``'to_departure_time'``: departure time of the trip at from the first stop after
      the screen line
    - ``'subshape_dist_frac'``: proportion of S's length at which the screen line
      intersects S

    Notes:

    - Assume the Feed's stop times DataFrame has an accurate ``shape_dist_traveled``
      column.
    - Assume that trips travel in the same direction as their shapes, an assumption
      that is part of the GTFS.
    - Assume that the screen line is straight and simple.
    - The algorithm works as follows

        1. Find the Feed's simple subshapes (computed via :func:`shapes.split_simple`)
           that intersect the screen lines.
        2. For each such subshape and screen line, compute the intersection points,
           the distance of each point along the subshape, aka the *crossing distance*,
           and the orientation of the screen line relative to the subshape.
        3. Restrict to trips active on the given dates and for each trip associated to
           an intersecting subshape above, interpolate a trip stop time
           for the intersection point using the crossing distance, subshape length,
           cumulative subshape length, and trip stop times.

    """
    from .shapes import split_simple

    # Convert geoms to UTM
    crs = screen_lines.estimate_utm_crs()
    screen_lines = screen_lines.to_crs(crs)

    # Create screen line IDs if necessary
    n = screen_lines.shape[0]
    if "screen_line_id" not in screen_lines.columns:
        screen_lines["screen_line_id"] = hp.make_ids(n, "sl")

    # Make a vector in the direction of each screen line to calculate crossing
    # orientation. Does not work in case of a bent screen line.
    p1 = screen_lines["geometry"].map(lambda x: np.array(x.coords[0]))
    p2 = screen_lines["geometry"].map(lambda x: np.array(x.coords[-1]))
    screen_lines["screen_line_vector"] = p2 - p1

    # Get the simple subshapes that intersect the screen lines.
    # Need subshapes to have only small gaps between them,
    # so `segmentize_m` needs to be small.
    subshapes = (
        feed.get_shapes(as_gdf=True, use_utm=True)
        .sjoin(screen_lines)
        .drop_duplicates("shape_id")
        .pipe(split_simple)
    )

    # Get intersection points of subshapes and screen lines
    g0 = (
        subshapes.sjoin(screen_lines.filter(["screen_line_id", "geometry"]))
        .merge(screen_lines, on="screen_line_id")
        .assign(
            int_point=lambda x: gpd.GeoSeries(x["geometry_x"], crs=crs).intersection(
                gpd.GeoSeries(x["geometry_y"], crs=crs)
            )
        )
    )

    # Unpack multipoint intersections to yield a new GeoDataFrame.
    # Should be very few multipoints.
    records = []
    for row in g0.itertuples(index=False):
        if isinstance(row.int_point, sg.Point):
            intersections = [row.int_point]
        else:
            intersections = row.int_point.geoms
        for int_point in intersections:
            record = {
                "subshape_id": row.subshape_id,
                "shape_id": row.shape_id,
                "subshape_length_m": row.subshape_length_m,
                "cum_length_m": row.cum_length_m,
                "screen_line_id": row.screen_line_id,
                "geometry": row.geometry_x,
                "int_point": int_point,
                "screen_line_vector": row.screen_line_vector,
            }
            records.append(record)

    g = gpd.GeoDataFrame.from_records(records).set_geometry("geometry").set_crs(crs)

    # Get distance (in meters) of each intersection point along subshape
    g["subshape_dist_frac"] = g.apply(
        lambda x: x["geometry"].project(x.int_point, normalized=True), axis=1
    )
    g["subshape_dist_m"] = g["subshape_dist_frac"] * g["subshape_length_m"]
    g["crossing_dist_m"] = (
        g["subshape_dist_m"] + g["cum_length_m"] - g["subshape_length_m"]
    )

    # Build a tiny vector along each subshape from the intersection point
    p2 = g.apply(
        lambda x: x["geometry"].interpolate(x["subshape_dist_m"] + 1), axis=1
    ).map(lambda x: np.array(x.coords[0]))
    p1 = g.int_point.map(lambda x: np.array(x.coords[0]))
    g["subshape_vector"] = p2 - p1

    # Compute crossing direction by taking the vector cross product of
    # the link vector and the screen line vector
    det = g.apply(
        lambda x: np.linalg.det(
            np.array([x["subshape_vector"], x["screen_line_vector"]])
        ),
        axis=1,
    )
    g["crossing_direction"] = det.map(lambda x: 1 if x >= 0 else -1)

    # Summarize work so far
    g = g[
        [
            "subshape_id",
            "shape_id",
            "screen_line_id",
            "subshape_dist_frac",
            "subshape_dist_m",
            "subshape_length_m",
            "crossing_direction",
            "crossing_dist_m",
        ]
    ]

    # Get stop times to compute crossing times
    feed = feed.convert_dist("m")
    frames = []
    for date in dates:
        st = (
            feed.get_stop_times(date)
            .pipe(_reshape_stop_times)
            .merge(feed.trips[["trip_id", "shape_id"]])
            # Keep only non-NaN departure times
            .loc[lambda x: x["from_departure_time"].notna()]
            .loc[lambda x: x["to_departure_time"].notna()]
            # Convert to seconds past midnight for upcoming crossing time calculation
            .assign(
                t1=lambda x: x["from_departure_time"].map(hp.timestr_to_seconds),
                t2=lambda x: x["to_departure_time"].map(hp.timestr_to_seconds),
            )
        )

        # Compute crossing times
        subframes = []
        for shape_id, group in g.groupby("shape_id"):
            f = (
                st.merge(group)
                # Only keep the times of the pair of stops on either side of each screen line,
                # whose distance along a trip shape is marked by column 'crossing_dist_m'
                .loc[lambda x: x["from_shape_dist_traveled"] <= x["crossing_dist_m"]]
                .loc[lambda x: x["crossing_dist_m"] <= x["to_shape_dist_traveled"]]
            )
            f["crossing_time"] = (
                f["t1"] + f["subshape_dist_frac"] * (f["t2"] - f["t1"])
            ).map(lambda x: hp.seconds_to_timestr(x))
            # Get distance along trip shape of crossing point
            subframes.append(f)

        if subframes:
            f = pd.concat(subframes).assign(date=date)
        else:
            f = pd.DataFrame()
        frames.append(f)

    f = pd.concat(frames)

    # Clean up
    final_cols = [
        "date",
        "segment_id",
        "segment_length",
        "screen_line_id",
        "shape_id",
        "trip_id",
        "direction_id",
        "route_id",
        "route_short_name",
        "route_type",
        "crossing_direction",
        "crossing_time",
        "crossing_dist_m",
    ]
    if include_testing_cols:
        final_cols += [
            "subshape_id",
            "subshape_length_m",
            "from_departure_time",
            "to_departure_time",
            "subshape_dist_frac",
            "subshape_dist_m",
        ]

    return (
        f
        # Append screen line info
        .merge(screen_lines.drop("geometry", axis=1))
        # Append extra trip info
        .merge(feed.trips[["trip_id", "direction_id", "route_id"]])
        .merge(feed.routes[["route_id", "route_short_name", "route_type"]])
        .filter(final_cols)
        .sort_values(
            ["screen_line_id", "trip_id", "crossing_dist_m"],
            ignore_index=True,
        )
    )
