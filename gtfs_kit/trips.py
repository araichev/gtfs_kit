"""
Functions about trips.
"""

from __future__ import annotations

import datetime as dt
import functools as ft
import json
from typing import TYPE_CHECKING, Iterable

import folium as fl
import folium.plugins as fp
import numpy as np
import pandas as pd
import shapely.geometry as sg
import shapely.ops as so

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_active_services(feed: "Feed", date: str) -> list[str]:
    """
    Given a Feed and a date string in YYYYMMDD format,
    return the list of service IDs that are active on the date.
    """
    active_services = set()
    removed_services = set()

    if feed.calendar is not None:
        # Convert date to weekday string (e.g., "monday")
        weekday_str = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()
        # Filter `calendar` to services active on date
        active_services |= set(
            feed.calendar.loc[
                lambda x: (x["start_date"] <= date)
                & (x["end_date"] >= date)
                & (x[weekday_str] == 1),
                "service_id",
            ]
        )

    if feed.calendar_dates is not None:
        # Filter `calendar_dates` to services added on date
        active_services |= set(
            feed.calendar_dates.loc[
                lambda x: (x["date"] == date) & (x["exception_type"] == 1),
                "service_id",
            ]
        )
        # Filter `calendar_dates` to services removed on date
        removed_services |= set(
            feed.calendar_dates.loc[
                lambda x: (x["date"] == date) & (x["exception_type"] == 2),
                "service_id",
            ]
        )

    return list(active_services - removed_services)


def get_trips(
    feed: "Feed",
    date: str | None = None,
    time: str | None = None,
    *,
    as_gdf: bool = False,
    use_utm: bool = False,
) -> pd.DataFrame:
    """
    Return ``feed.trips``.
    If date (YYYYMMDD date string) is given then subset the result to trips
    that start on that date.
    If a time (HH:MM:SS string, possibly with HH > 23) is given in addition to a date,
    then further subset the result to trips in service at that time.

    If ``as_gdf`` and ``feed.shapes`` is not None, then return the trips as a
    GeoDataFrame of LineStrings representating trip shapes.
    Use local UTM CRS if ``use_utm``; otherwise it the WGS84 CRS.
    If ``as_gdf`` and ``feed.shapes`` is ``None``, then raise a ValueError.
    """
    if feed.trips is None:
        return None

    f = feed.trips
    if date is not None:
        f = f.loc[lambda x: x["service_id"].isin(get_active_services(feed, date))]

        if time is not None:
            # Get trips active during given time
            def get_active(group):
                d = {}
                start = group["departure_time"].dropna().min()
                end = group["departure_time"].dropna().max()
                try:
                    result = start <= time <= end
                except TypeError:
                    result = False
                d["is_active"] = result
                return pd.Series(d)

            f = (
                f.merge(feed.stop_times[["trip_id", "departure_time"]])
                .groupby("trip_id")
                .apply(get_active, include_groups=False)
                .reset_index()
                .loc[lambda x: x["is_active"]]
                .merge(f)
                .drop("is_active", axis=1)
            )

    if as_gdf:
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")
        else:
            from .shapes import get_shapes

            f = (
                get_shapes(feed, as_gdf=True, use_utm=use_utm)
                .filter(["shape_id", "geometry"])
                .merge(f, how="right")
                .filter(f.columns.tolist() + ["geometry"])
            )

    return f


def compute_trip_activity(feed: "Feed", dates: list[str]) -> pd.DataFrame:
    """
    Mark trips as active or inactive on the given dates (YYYYMMDD date strings).
    Return a table with the columns

    - ``'trip_id'``
    - ``dates[0]``: 1 if the trip is active on ``dates[0]``;
      0 otherwise
    - ``dates[1]``: 1 if the trip is active on ``dates[1]``;
      0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the trip is active on ``dates[-1]``;
      0 otherwise

    If ``dates`` is ``None`` or the empty list, then return an
    empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    active_services_by_date = {date: get_active_services(feed, date) for date in dates}

    f = feed.trips[["trip_id", "service_id"]].copy()
    for date in dates:
        active_services = active_services_by_date[date]
        f[date] = f["service_id"].isin(active_services).astype(int)
    f.drop("service_id", axis=1, inplace=True)

    return f


def compute_busiest_date(feed: "Feed", dates: list[str]) -> str:
    """
    Given a list of dates (YYYYMMDD date strings), return the first date that has the
    maximum number of active trips.
    """
    f = feed.compute_trip_activity(dates)
    s = [(f[c].sum(), c) for c in f.columns if c != "trip_id"]
    return max(s)[1]


def name_stop_patterns(feed: "Feed") -> pd.DataFrame:
    """
    For each (route ID, direction ID) pair, find the distinct stop patterns of its
    trips, and assign them each an integer *pattern rank* based on the stop pattern's
    frequency rank, where 1 is the most frequent stop pattern, 2 is the second most
    frequent, etc.
    Return the DataFrame ``feed.trips`` with the additional column
    ``stop_pattern_name``, which equals the trip's 'direction_id' concatenated with a
    dash and its stop pattern rank.

    If ``feed.trips`` has no 'direction_id' column, then temporarily create one equal
    to all zeros, proceed as above, then delete the column.
    """
    # Collect stop patterns
    trips = feed.trips.copy()
    if "direction_id" in trips.columns:
        has_direction = True
    else:
        has_direction = False
        trips["direction_id"] = 0  # placeholder to ease calculations below
    f = (
        feed.stop_times.sort_values(["trip_id", "stop_sequence"])
        .groupby("trip_id")
        .apply(
            lambda x: pd.Series({"stop_pattern": "-".join(x["stop_id"])}),
            include_groups=False,
        )
        .reset_index()
        .merge(trips[["route_id", "trip_id", "direction_id"]])
    )
    # Compute pattern ranks
    f["rank"] = f.groupby(["route_id", "direction_id"])["stop_pattern"].transform(
        lambda x: pd.Series(x).map(
            {pattern: idx + 1 for idx, pattern in enumerate(x.value_counts().index)}
        )
    )
    # Assign pattern names
    f["stop_pattern_name"] = (
        f["direction_id"].astype(str).str.cat(f["rank"].astype(str), sep="-")
    )
    ff = trips.merge(f[["route_id", "trip_id", "direction_id", "stop_pattern_name"]])
    # Clean up
    if not has_direction:
        del ff["direction_id"]

    return ff


def compute_trip_stats(
    feed: "Feed",
    route_ids: list[str | None] = None,
    *,
    compute_dist_from_shapes: bool = False,
) -> pd.DataFrame:
    """
    Return a DataFrame with the following columns:

    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: NaN if missing from feed
    - ``'shape_id'``: NaN if missing from feed
    - ``'stop_pattern_name'``: output from :func:`name_stop_patterns`
    - ``'num_stops'``: number of stops on trip
    - ``'start_time'``: first departure time of the trip
    - ``'end_time'``: last departure time of the trip
    - ``'start_stop_id'``: stop ID of the first stop of the trip
    - ``'end_stop_id'``: stop ID of the last stop of the trip
    - ``'is_loop'``: 1 if the start and end stop are less than 400m apart and
      0 otherwise
    - ``'distance'``: distance of the trip;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'duration'``: duration of the trip in hours
    - ``'speed'``: distance/duration

    If ``feed.stop_times`` has a ``shape_dist_traveled`` column with at
    least one non-NaN value and ``compute_dist_from_shapes == False``,
    then use that column to compute the distance column.
    Else if ``feed.shapes is not None``, then compute the distance
    column using the shapes and Shapely.
    Otherwise, set the distances to NaN.

    If route IDs are given, then restrict to trips on those routes.

    Notes
    -----
    - Assume the following feed attributes are not ``None``:

        * ``feed.trips``
        * ``feed.routes``
        * ``feed.stop_times``
        * ``feed.shapes`` (optionally)

    - Calculating trip distances with ``compute_dist_from_shapes=True``
      seems pretty accurate.  For example, calculating trip distances on
      `this Portland feed
      <https://transitfeeds.com/p/trimet/43/1400947517>`_
      using ``compute_dist_from_shapes=False`` and
      ``compute_dist_from_shapes=True``,
      yields a difference of at most 0.83km from the original values.

    """
    f = name_stop_patterns(feed)

    # Restrict to given route IDs
    if route_ids is not None:
        f = f[f["route_id"].isin(route_ids)].copy()

    # Merge with stop times and extra trip info.
    # Convert departure times to seconds past midnight to
    # compute trip durations later.
    if "direction_id" not in f.columns:
        f["direction_id"] = np.nan
    if "shape_id" not in f.columns:
        f["shape_id"] = np.nan

    f = (
        f[["route_id", "trip_id", "direction_id", "shape_id", "stop_pattern_name"]]
        .merge(feed.routes[["route_id", "route_short_name", "route_type"]])
        .merge(feed.stop_times)
        .sort_values(["trip_id", "stop_sequence"])
        .assign(departure_time=lambda x: x["departure_time"].map(hp.timestr_to_seconds))
    )

    # Compute all trips stats except distance,
    # which is possibly more involved
    geometry_by_stop = feed.build_geometry_by_stop(use_utm=True)
    g = f.groupby("trip_id")

    def my_agg(group):
        d = dict()
        d["route_id"] = group["route_id"].iat[0]
        d["route_short_name"] = group["route_short_name"].iat[0]
        d["route_type"] = group["route_type"].iat[0]
        d["direction_id"] = group["direction_id"].iat[0]
        d["shape_id"] = group["shape_id"].iat[0]
        d["stop_pattern_name"] = group["stop_pattern_name"].iat[0]
        d["num_stops"] = group.shape[0]
        d["start_time"] = group["departure_time"].iat[0]
        d["end_time"] = group["departure_time"].iat[-1]
        d["start_stop_id"] = group["stop_id"].iat[0]
        d["end_stop_id"] = group["stop_id"].iat[-1]
        dist = geometry_by_stop[d["start_stop_id"]].distance(
            geometry_by_stop[d["end_stop_id"]]
        )
        d["is_loop"] = int(dist < 400)
        d["duration"] = (d["end_time"] - d["start_time"]) / 3600
        return pd.Series(d)

    # Apply my_agg, but don't reset index yet.
    # Need trip ID as index to line up the results of the
    # forthcoming distance calculation
    h = g.apply(my_agg, include_groups=False)

    # Compute distance
    if hp.is_not_null(f, "shape_dist_traveled") and not compute_dist_from_shapes:
        # Compute distances using shape_dist_traveled column, converting to km or mi
        if hp.is_metric(feed.dist_units):
            convert_dist = hp.get_convert_dist(feed.dist_units, "km")
        else:
            convert_dist = hp.get_convert_dist(feed.dist_units, "mi")
        h["distance"] = g.apply(
            lambda group: convert_dist(group["shape_dist_traveled"].max()),
            include_groups=False,
        )
    elif feed.shapes is not None:
        # Compute distances using the shapes and Shapely
        geometry_by_shape = feed.build_geometry_by_shape(use_utm=True)
        # Convert to km or mi
        if hp.is_metric(feed.dist_units):
            m_to_dist = hp.get_convert_dist("m", "km")
        else:
            m_to_dist = hp.get_convert_dist("m", "mi")

        def compute_dist(group):
            """
            Return the distance traveled along the trip between the
            first and last stops.
            If that distance is negative or if the trip's linestring
            intersects itfeed, then return the length of the trip's
            linestring instead.
            """
            shape = group["shape_id"].iat[0]
            try:
                # Get the linestring for this trip
                linestring = geometry_by_shape[shape]
            except KeyError:
                # Shape ID is NaN or doesn't exist in shapes.
                # No can do.
                return np.nan

            # If the linestring intersects itfeed, then that can cause
            # errors in the computation below, so just
            # return the length of the linestring as a good approximation
            D = linestring.length
            if not linestring.is_simple:
                return D

            # Otherwise, return the difference of the distances along
            # the linestring of the first and last stop
            start_stop = group["stop_id"].iat[0]
            end_stop = group["stop_id"].iat[-1]
            try:
                start_point = geometry_by_stop[start_stop]
                end_point = geometry_by_stop[end_stop]
            except KeyError:
                # One of the two stop IDs is NaN, so just
                # return the length of the linestring
                return D
            d1 = linestring.project(start_point)
            d2 = linestring.project(end_point)
            d = d2 - d1
            if 0 < d < D + 100:
                return d
            else:
                # Something is probably wrong, so just
                # return the length of the linestring
                return D

        h["distance"] = g.apply(compute_dist, include_groups=False)
        # Convert from meters
        h["distance"] = h["distance"].map(m_to_dist)
    else:
        h["distance"] = np.nan

    # Reset index and compute final stats
    h = h.reset_index()
    h["speed"] = h["distance"] / h["duration"]
    h[["start_time", "end_time"]] = h[["start_time", "end_time"]].map(
        lambda x: hp.seconds_to_timestr(x)
    )

    return h.sort_values(["route_id", "direction_id", "start_time"], ignore_index=True)


def locate_trips(feed: "Feed", date: str, times: list[str]) -> pd.DataFrame:
    """
    Return the positions of all trips active on the
    given date (YYYYMMDD date string) and times (HH:MM:SS time strings,
    possibly with HH > 23).

    Return a DataFrame with the columns

    - ``'trip_id'``
    - ``'route_id'``
    - ``'direction_id'``: all NaNs if ``feed.trips.direction_id`` is
      missing
    - ``'time'``
    - ``'rel_dist'``: number between 0 (start) and 1 (end)
      indicating the relative distance of the trip along its path
    - ``'lon'``: longitude of trip at given time
    - ``'lat'``: latitude of trip at given time

    Assume ``feed.stop_times`` has an accurate
    ``shape_dist_traveled`` column.
    """
    if not hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        raise ValueError(
            "feed.stop_times needs to have a non-null shape_dist_traveled "
            "column. You can create it, possibly with some inaccuracies, "
            "via feed2 = feed.append_dist_to_stop_times()."
        )

    if "shape_id" not in feed.trips.columns:
        raise ValueError("feed.trips.shape_id must exist.")

    # Start with stop times active on date
    f = feed.get_stop_times(date)
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)

    # Compute relative distance of each trip along its path
    # at the given time times.
    # Use linear interpolation based on stop departure times and
    # shape distance traveled.
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
    sample_times = np.array([hp.timestr_to_seconds(s) for s in times])

    def compute_rel_dist(group):
        dists = sorted(group["shape_dist_traveled"].values)
        times = sorted(group["departure_time"].values)
        ts = sample_times[(sample_times >= times[0]) & (sample_times <= times[-1])]
        ds = np.interp(ts, times, dists)
        return pd.DataFrame({"time": ts, "rel_dist": ds / dists[-1]})

    # return f.groupby('trip_id', group_keys=False).\
    #   apply(compute_rel_dist).reset_index()
    g = f.groupby("trip_id").apply(compute_rel_dist, include_groups=False).reset_index()

    # Delete extraneous multi-index column
    del g["level_1"]

    # Convert times back to time strings
    g["time"] = g["time"].map(lambda x: hp.seconds_to_timestr(x))

    # Merge in more trip info and
    # compute longitude and latitude of trip from relative distance
    t = feed.trips.copy()
    if "direction_id" not in t.columns:
        t["direction_id"] = np.nan

    h = pd.merge(g, t[["trip_id", "route_id", "direction_id", "shape_id"]])
    if not h.shape[0]:
        # Return a DataFrame with the promised headers but no data.
        # Without this check, result below could be an empty DataFrame.
        h["lon"] = pd.Series()
        h["lat"] = pd.Series()
        return h

    def get_lonlat(group):
        shape = group.name
        linestring = geometry_by_shape[shape]
        lonlats = [
            linestring.interpolate(d, normalized=True).coords[0]
            for d in group["rel_dist"].values
        ]
        group["lon"], group["lat"] = zip(*lonlats)
        return group

    return (
        h.groupby("shape_id")
        .apply(get_lonlat, include_groups=False)
        .reset_index()
        .drop("level_1", axis=1)  # Where did this column come from?
    )


def trips_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str] | None = None,
    *,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection (in WGS84 coordinates) of LineString features
    representing this Feed's trips.

    If an iterable of trip IDs is given, then subset to those trips, which could yield
    an empty FeatureCollection in case all trip IDs are invalid.
    If ``include_stops``, then include the trip stops as Point features.
    If the Feed has no shapes, then raise a ValueError.
    """
    g = get_trips(feed, as_gdf=True)
    if trip_ids is not None:
        g = g.loc[lambda x: x["trip_id"].isin(trip_ids)]

    if g is None or g.empty:
        gj = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        gj = json.loads(g.to_json(drop_id=True))
        if include_stops:
            st_gj = feed.stop_times_to_geojson(trip_ids)
            gj["features"].extend(st_gj["features"])

    return gj


def map_trips(
    feed: "Feed",
    trip_ids: Iterable[str],
    color_palette: list[str] = cs.COLORS_SET2,
    *,
    show_stops: bool = False,
    show_direction: bool = False,
):
    """
    Return a Folium map showing the given trips and (optionally)
    their stops.
    If any of the given trip IDs are not found in the feed, then raise a ValueError.
    If ``include_direction``, then use the Folium plugin PolyLineTextPath to draw arrows
    on each trip polyline indicating its direction of travel; this fails to work in some
    browsers, such as Brave 0.68.132.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Create colors
    n = len(trip_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, trip_id in enumerate(trip_ids):
        collection = trips_to_geojson(feed, [trip_id], include_stops=show_stops)

        group = fl.FeatureGroup(name=f"Trip {trip_id}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop if present
            if f["geometry"]["type"] == "Point":
                lon, lat = f["geometry"]["coordinates"]
                fl.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    fill=True,
                    color=color,
                    weight=1,
                    popup=fl.Popup(hp.make_html(prop)),
                ).add_to(group)

            # Add trip
            else:
                path = fl.PolyLine(
                    [[x[1], x[0]] for x in f["geometry"]["coordinates"]],
                    color=color,
                    popup=hp.make_html(prop),
                )

                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

                if show_direction:
                    # Direction arrows, assuming, as GTFS does, that
                    # trip direction equals LineString direction
                    fp.PolyLineTextPath(
                        path,
                        "        \u27a4        ",
                        repeat=True,
                        offset=5.5,
                        attributes={"fill": color, "font-size": "18"},
                    ).add_to(group)

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    # Folium wants a different ordering
    bounds = [(bounds[1], bounds[0]), (bounds[3], bounds[2])]
    my_map.fit_bounds(bounds)

    return my_map
