"""
Functions about stops.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Iterable

import folium as fl
import folium.plugins as fp
import geopandas as gpd
import numpy as np
import pandas as pd

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


#: Leaflet circleMarker parameters for mapping stops
STOP_STYLE = {
    "radius": 8,
    "fill": "true",
    "color": cs.COLORS_SET2[1],
    "weight": 1,
    "fillOpacity": 0.75,
}


def geometrize_stops(stops: pd.DataFrame, *, use_utm: bool = False) -> gpd.GeoDataFrame:
    """
    Given a stops DataFrame, convert it to a GeoPandas GeoDataFrame of Points
    and return the result, which will no longer have the columns ``'stop_lon'`` and
    ``'stop_lat'``.
    """
    g = (
        stops.assign(geometry=gpd.points_from_xy(x=stops.stop_lon, y=stops.stop_lat))
        .drop(["stop_lon", "stop_lat"], axis=1)
        .pipe(gpd.GeoDataFrame, crs=cs.WGS84)
    )

    if use_utm:
        g = g.to_crs(g.estimate_utm_crs())

    return g


def ungeometrize_stops(stops_g: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    The inverse of :func:`geometrize_stops`.

    If ``stops_g`` is in UTM coordinates (has a UTM CRS property),
    then convert those UTM coordinates back to WGS84 coordinates,
    which is the standard for a GTFS shapes table.
    """
    f = stops_g.copy().to_crs(cs.WGS84)
    f["stop_lon"], f["stop_lat"] = zip(*f["geometry"].map(lambda p: [p.x, p.y]))
    del f["geometry"]
    return f


def get_stops(
    feed: "Feed",
    date: str | None = None,
    trip_ids: Iterable[str] | None = None,
    route_ids: Iterable[str] | None = None,
    *,
    in_stations: bool = False,
    as_gdf: bool = False,
    use_utm: bool = False,
) -> pd.DataFrame:
    """
    Return ``feed.stops``.
    If a YYYYMMDD date string is given, then subset to stops
    active (visited by trips) on that date.
    If trip IDs are given, then subset further to stops visited by those
    trips.
    If route IDs are given, then ignore the trip IDs and subset further
    to stops visited by those routes.
    If ``in_stations``, then subset further stops in stations if station data
    is available.
    If ``as_gdf``, then return the result as a GeoDataFrame with a 'geometry'
    column of points instead of 'stop_lat' and 'stop_lon' columns.
    The GeoDataFrame will have a UTM CRS if ``use_utm`` and a WGS84 CRS otherwise.
    """
    s = feed.stops.copy()
    if date is not None:
        A = feed.get_stop_times(date).stop_id
        s = s.loc[lambda x: x.stop_id.isin(A)].copy()
    if trip_ids is not None:
        st = feed.stop_times.copy()
        B = st.loc[lambda x: x.trip_id.isin(trip_ids), "stop_id"].copy()
        s = s.loc[lambda x: x.stop_id.isin(B)].copy()
    elif route_ids is not None:
        A = feed.trips.loc[lambda x: x.route_id.isin(route_ids), "trip_id"].copy()
        st = feed.stop_times.copy()
        B = st.loc[lambda x: x.trip_id.isin(A), "stop_id"].copy()
        s = s.loc[lambda x: x.stop_id.isin(B)].copy()
    if in_stations and set(["location_type", "parent_station"]) <= set(s.columns):
        s = s.loc[lambda x: (x.location_type != 1) & (x.parent_station.notna())].copy()
    if as_gdf:
        s = geometrize_stops(s, use_utm=use_utm)
    return s


def compute_stop_activity(feed: "Feed", dates: list[str]) -> pd.DataFrame:
    """
    Mark stops as active or inactive on the given dates (YYYYMMDD date strings).
    A stop is *active* on a given date if some trips that starts on the
    date visits the stop (possibly after midnight).

    Return a DataFrame with the columns

    - stop_id
    - ``dates[0]``: 1 if the stop has at least one trip visiting it
      on ``dates[0]``; 0 otherwise
    - ``dates[1]``: 1 if the stop has at least one trip visiting it
      on ``dates[1]``; 0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the stop has at least one trip visiting it
      on ``dates[-1]``; 0 otherwise

    If all dates lie outside the Feed period, then return an empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    trip_activity = feed.compute_trip_activity(dates)
    g = pd.merge(trip_activity, feed.stop_times).groupby("stop_id")
    # Pandas won't allow me to simply return g[dates].max().reset_index().
    # I get ``TypeError: unorderable types: datetime.date() < str()``.
    # So here's a workaround.
    for i, date in enumerate(dates):
        if i == 0:
            f = g[date].max().reset_index()
        else:
            f = f.merge(g[date].max().reset_index())
    return f


def compute_stop_stats_0(
    stop_times_subset: pd.DataFrame,
    trips_subset: pd.DataFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Given a subset of a trips DataFrame and a subset of a stop times
    DataFrame, return a DataFrame that provides summary stats about the
    stops in the inner join of the two DataFrames.

    If ``split_directions``, then separate the stop stats by direction (0 or 1)
    of the trips visiting the stops.
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - stop_id
    - direction_id: present if and only if ``split_directions``
    - num_routes: number of routes visiting stop
      (in the given direction)
    - num_trips: number of trips visiting stop
      (in the givin direction)
    - max_headway: maximum of the durations (in minutes)
      between trip departures at the stop between
      ``headway_start_time`` and ``headway_end_time``
    - min_headway: minimum of the durations (in minutes) mentioned
      above
    - mean_headway: mean of the durations (in minutes) mentioned
      above
    - start_time: earliest departure time of a trip from this stop
    - end_time: latest departure time of a trip from this stop

    Notes
    -----
    - If ``trips_subset`` is empty, then return an empty DataFrame.
    - Raise a ValueError if ``split_directions`` and no non-NaN
      direction ID values present.

    """
    final_cols = [
        "stop_id",
        "num_trips",
        "num_routes",
        "max_headway",  # minutes
        "min_headway",  # minutes
        "mean_headway",  # minutes
        "start_time",  # HH:MM:SS
        "end_time",  # HH:MM:SS
    ]
    if split_directions:
        final_cols.append("direction_id")

    null_stats = pd.DataFrame(data=[], columns=final_cols)

    # Handle defunct case
    if trips_subset.empty:
        return null_stats

    f = stop_times_subset.merge(trips_subset)

    # Convert departure times to seconds to ease headway calculations
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    # Compute stats for each stop
    def agg(group):
        # Operate on the group of all stop times for an individual stop
        d = dict()
        d["num_routes"] = group["route_id"].unique().size
        d["num_trips"] = group.shape[0]
        d["start_time"] = group["departure_time"].min()
        d["end_time"] = group["departure_time"].max()
        headways = []
        dtimes = sorted(
            [
                dtime
                for dtime in group["departure_time"].values
                if headway_start <= dtime <= headway_end
            ]
        )
        headways.extend([dtimes[i + 1] - dtimes[i] for i in range(len(dtimes) - 1)])
        if headways:
            d["max_headway"] = np.max(headways) / 60  # minutes
            d["min_headway"] = np.min(headways) / 60  # minutes
            d["mean_headway"] = np.mean(headways) / 60  # minutes
        else:
            d["max_headway"] = np.nan
            d["min_headway"] = np.nan
            d["mean_headway"] = np.nan
        return pd.Series(d)

    if split_directions:
        if "direction_id" not in f.columns:
            f["direction_id"] = np.nan
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError("At least one trip direction ID value must be non-NaN.")
        g = f.groupby(["stop_id", "direction_id"])
    else:
        g = f.groupby("stop_id")

    result = g.apply(agg, include_groups=False).reset_index()

    # Convert start and end times to time strings
    result[["start_time", "end_time"]] = result[["start_time", "end_time"]].map(
        lambda x: hp.seconds_to_timestr(x)
    )

    return result.filter(final_cols)


def compute_stop_stats(
    feed: "Feed",
    dates: list[str],
    stop_ids: list[str | None] = None,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats for all stops for the given dates (YYYYMMDD date strings).
    Optionally, restrict to the stop IDs given.

    If ``split_directions``, then separate the stop stats by direction (0 or 1)
    of the trips visiting the stops.
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'date'``
    - ``'stop_id'``
    - ``'direction_id'``: present if and only if ``split_directions``
    - ``'num_routes'``: number of routes visiting the stop
      (in the given direction) on the date
    - ``'num_trips'``: number of trips visiting stop
      (in the givin direction) on the date
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip departures at the stop between
      ``headway_start_time`` and ``headway_end_time`` on the date
    - ``'min_headway'``: minimum of the durations (in minutes) mentioned
      above
    - ``'mean_headway'``: mean of the durations (in minutes) mentioned
      above
    - ``'start_time'``: earliest departure time of a trip from this stop
      on the date
    - ``'end_time'``: latest departure time of a trip from this stop on
      the date

    Exclude dates with no active stops, which could yield the empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    null_stats = compute_stop_stats_0(
        pd.DataFrame(), pd.DataFrame(), split_directions=split_directions
    )
    final_cols = ["date"] + null_stats.columns.tolist()

    # Handle defunct case
    if not dates:
        return null_stats

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFarme.
    stats_by_ids = {}

    # Restrict stop times to stop IDs if specified
    if stop_ids is not None:
        stop_times_subset = feed.stop_times.loc[
            lambda x: x["stop_id"].isin(stop_ids)
        ].copy()
    else:
        stop_times_subset = feed.stop_times.copy()

    activity = feed.compute_trip_activity(dates)

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
            t = feed.trips
            trips = t[t["trip_id"].isin(ids)].copy()
            stats = (
                compute_stop_stats_0(
                    stop_times_subset,
                    trips,
                    split_directions=split_directions,
                    headway_start_time=headway_start_time,
                    headway_end_time=headway_end_time,
                )
                # Assign date
                .assign(date=date)
            )
            # Memoize stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Assemble stats into a single DataFrame
    return pd.concat(frames).filter(final_cols)


def compute_stop_time_series_0(
    stop_times_subset: pd.DataFrame,
    trips_subset: pd.DataFrame,
    freq: str = "h",
    date_label: str = "20010101",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stop stats in a 24-hour time series form at the given Pandas frequency
    for stops in the inner join of the given subset of stop times and trips.

    If ``split_directions``, then separate each stop's stats by trip direction.
    Use the given YYYYMMDD date label as the date in the time series index.

    Return a long-format DataFrame with columns

    - ``datetime``: datetime object for the given date and frequency chunks
    - ``stop_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: the number of trips that visit the stop in the time bin and
      have a nonnull departure time from the stop

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Stop times with null departure times are ignored, so the aggregate
      of ``num_trips`` across the day could be less than the
      ``num_trips`` column in :func:`compute_stop_stats_0`
    - All trip departure times are taken modulo 24 hours,
      so routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series.
    - 'num_trips' should be resampled with ``how=np.sum``
    - If ``trips_subset`` is empty, then return an empty DataFrame
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    final_cols = [
        "datetime",
        "stop_id",
        "num_trips",
    ]
    if split_directions:
        final_cols.insert(2, "direction_id")
    null_stats = pd.DataFrame([], columns=final_cols)

    # Handle defunct case
    if trips_subset.empty:
        return null_stats

    f = pd.merge(stop_times_subset, trips_subset)

    if split_directions:
        if "direction_id" not in f.columns:
            f["direction_id"] = pd.NA
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.empty:
            raise ValueError("At least one trip direction ID value must be non-NaN.")

        # Alter stop IDs to encode trip direction:
        # <stop ID>-0 and <stop ID>-1
        f["stop_id"] = f["stop_id"] + "-" + f["direction_id"].map(str)

    # Bin each stop departure time
    bins = [i for i in range(24 * 60)]  # One bin for each minute
    num_bins = len(bins)

    def timestr_to_min(x):
        return hp.timestr_to_seconds(x, mod24=True) // 60

    f["departure_index"] = f["departure_time"].map(timestr_to_min)

    # Create one time series for each stop
    series_by_stop = {
        stop: [0 for i in range(num_bins)] for stop in f["stop_id"].unique()
    }

    for stop, group in f.groupby("stop_id"):
        counts = Counter((bin, 0) for bin in bins) + Counter(
            group["departure_index"].values
        )
        series_by_stop[stop] = [counts[bin] for bin in bins]

    # Build per-indicator DataFrames indexed by minute across the provided date
    rng = pd.date_range(
        pd.to_datetime(f"{date_label} 00:00:00"), periods=24 * 60, freq="Min"
    )
    series_by_indicator = {"num_trips": pd.DataFrame(series_by_stop, index=rng).fillna(0)}

    # Combine into a single long-form time series per route (and direction if requested);
    # hp.combine_time_series is expected to compute derived fields like service_speed
    g = hp.combine_time_series(
        series_by_indicator, kind="stop", split_directions=split_directions
    )
    # Downsample to requested frequency (sum for counts/durations/distances; speed handled by helper)
    return hp.downsample(g, freq=freq)


def compute_stop_time_series(
    feed: "Feed",
    dates: list[str],
    stop_ids: list[str | None] = None,
    freq: str = "h",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute time series for the given stops (defaults to all stops in Feed)
    on the given dates (YYYYMMDD date strings) at the
    given frequency (Pandas frequency string, e.g. ``'5Min'``).
    Return a long-format DataFrame with columns

    - ``datetime``: datetime object for the given date and frequency chunks
    - ``stop_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: the number of trips that visit the stop in the time bin and
      have a nonnull departure time from the stop

    Exclude dates that lie outside of the Feed's date range.
    If all dates lie outside the Feed's date range, then return an
    empty DataFrame

    If ``split_directions``, then separate the stop stats by direction (0 or 1)
    of the trips visiting the stops.

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Stop times with null departure times are ignored, so the aggregate
      of ``num_trips`` across the day could be less than the
      ``num_trips`` column in :func:`compute_stop_stats_0`
    - All trip departure times are taken modulo 24 hours,
      so routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series.
    - 'num_trips' should be resampled with ``how=np.sum``
    - If ``trips_subset`` is empty, then return an empty DataFrame
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    dates = feed.subset_dates(dates)
    null_stats = compute_stop_time_series_0(
        pd.DataFrame(), pd.DataFrame(), split_directions=split_directions
    )
    # Handle defunct case
    if not dates:
        return null_stats

    activity = feed.compute_trip_activity(dates)

    # Restrict stop times to stop IDs if specified
    if stop_ids is not None:
        stop_times_subset = feed.stop_times.loc[
            lambda x: x["stop_id"].isin(stop_ids)
        ].copy()
    else:
        stop_times_subset = feed.stop_times.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary recomputations.
    # Store in dictionary of the form
    # trip ID sequence -> stats table
    frames = []
    stats_by_ids = {}
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            # Reuse stats with updated date
            stats = stats_by_ids[ids].pipe(hp.replace_date, date=date)
        elif ids:
            # Compute stats afresh
            trips_subset = feed.trips.loc[lambda x: x["trip_id"].isin(ids)].copy()
            stats = compute_stop_time_series_0(
                stop_times_subset,
                trips_subset,
                split_directions=split_directions,
                freq=freq,
                date_label=date,
            ).pipe(hp.replace_date, date=date)
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    return pd.concat(frames, ignore_index=True)


def build_stop_timetable(feed: "Feed", stop_id: str, dates: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame containing the timetable for the given stop ID
    and dates (YYYYMMDD date strings)

    Return a DataFrame whose columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times`` plus ``'date'``, and the stop IDs are restricted to the given
    stop ID.
    The result is sorted by date then departure time.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    st = feed.stop_times[feed.stop_times["stop_id"] == stop_id]
    t = pd.merge(feed.trips, st, on="trip_id")

    a = feed.compute_trip_activity(dates)
    a = a[a["trip_id"].isin(t["trip_id"])]

    a_long = a.melt(
        id_vars="trip_id",
        value_vars=dates,
        var_name="date",
        value_name="active",
    )
    a_long = a_long[a_long["active"] == 1][["trip_id", "date"]]

    return (
        pd.merge(t, a_long)
        .assign(dtime=lambda x: x["departure_time"].map(hp.timestr_to_seconds))
        .sort_values(["date", "dtime"], ignore_index=True)
        .drop("dtime", axis=1)
    )


def build_geometry_by_stop(
    feed: "Feed", stop_ids: Iterable[str] | None = None, *, use_utm: bool = False
) -> dict:
    """
    Return a dictionary of the form <stop ID> -> <Shapely Point representing stop>.
    """
    g = get_stops(feed, as_gdf=True, use_utm=use_utm)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]
    return dict(g.filter(["stop_id", "geometry"]).values)


def stops_to_geojson(feed: "Feed", stop_ids: Iterable[str | None] = None) -> dict:
    """
    Return a GeoJSON FeatureCollection of Point features
    representing all the stops in ``feed.stops``.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If an iterable of stop IDs is given, then subset to those stops.
    If some of the given stop IDs are not found in the feed, then raise a ValueError.
    """
    g = get_stops(feed, as_gdf=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return json.loads(g.to_json(drop_id=True))


def get_stops_in_area(
    feed: "Feed",
    area: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Return the subset of ``feed.stops`` that contains all stops that lie
    within the given GeoDataFrame of polygons.
    """
    return (
        gpd.sjoin(get_stops(feed, as_gdf=True), area.to_crs(cs.WGS84))
        .filter(["stop_id"])
        .merge(feed.stops)
    )


def map_stops(feed: "Feed", stop_ids: Iterable[str], stop_style: dict = STOP_STYLE):
    """
    Return a Folium map showing the given stops of this Feed.
    If some of the given stop IDs are not found in the feed, then raise a ValueError.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Add stops to feature group
    stops = feed.stops.loc[lambda x: x.stop_id.isin(stop_ids)].fillna("n/a")

    # Add stops with clustering
    callback = f"""\
    function (row) {{
        var imarker;
        marker = L.circleMarker(new L.LatLng(row[0], row[1]),
            {stop_style}
        );
        marker.bindPopup(
            '<b>Stop name</b>: ' + row[2] + '<br>' +
            '<b>Stop code</b>: ' + row[3] + '<br>' +
            '<b>Stop ID</b>: ' + row[4]
        );
        return marker;
    }};
    """
    fp.FastMarkerCluster(
        data=stops[
            ["stop_lat", "stop_lon", "stop_name", "stop_code", "stop_id"]
        ].values.tolist(),
        callback=callback,
        disableClusteringAtZoom=14,
    ).add_to(my_map)

    # Fit map to stop bounds
    bounds = [
        (stops.stop_lat.min(), stops.stop_lon.min()),
        (stops.stop_lat.max(), stops.stop_lon.max()),
    ]
    my_map.fit_bounds(bounds, padding=[1, 1])

    return my_map
