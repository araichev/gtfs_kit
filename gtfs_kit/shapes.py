"""
Functions about shapes.
"""

from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import json

import geopandas as gpd
import pandas as pd
import shapely
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def append_dist_to_shapes(feed: "Feed") -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` field in
    ``feed.shapes`` in terms of the distance units ``feed.dist_units``.
    Return the resulting Feed.

    As a benchmark, using this function on `this Portland feed
    <https://transitfeeds.com/p/trimet/43/1400947517>`_
    produces a ``shape_dist_traveled`` column that differs by at most
    0.016 km in absolute value from of the original values.
    """
    if feed.shapes is None:
        raise ValueError("This function requires the feed to have a shapes.txt file")

    feed = feed.copy()

    g = (
        feed.shapes.assign(
            geometry=lambda x: gpd.points_from_xy(x["shape_pt_lon"], x["shape_pt_lat"])
        )
        .pipe(gpd.GeoDataFrame, crs=cs.WGS84)
        .pipe(lambda x: x.to_crs(x.estimate_utm_crs()))
        .sort_values(["shape_id", "shape_pt_sequence"])
    )
    # Compute cumulative between successive points within shape
    g["prev_geom"] = g.groupby("shape_id")["geometry"].shift(1)
    g["dist_m"] = g.apply(
        lambda row: (
            row["geometry"].distance(row["prev_geom"])
            if pd.notnull(row["prev_geom"])
            else 0
        ),
        axis=1,
    )
    g["shape_dist_traveled"] = (
        g.groupby("shape_id")["dist_m"]
        .cumsum()
        .map(hp.get_convert_dist("m", feed.dist_units))
    )

    feed.shapes = g.drop(["geometry", "prev_geom", "dist_m"], axis=1)
    return feed


def geometrize_shapes(shapes: pd.DataFrame, *, use_utm: bool = False) -> gpd.GeoDataFrame:
    """
    Given a GTFS shapes DataFrame, convert it to a GeoDataFrame of LineStrings
    and return the result, which will no longer have the columns
    ``'shape_pt_sequence'``, ``'shape_pt_lon'``,
    ``'shape_pt_lat'``, and ``'shape_dist_traveled'``.

    If ``use_utm``, then use local UTM coordinates for the geometries.
    """

    def my_agg(group):
        d = {}
        coords = group[["shape_pt_lon", "shape_pt_lat"]].values
        try:
            d["geometry"] = sg.LineString(coords)
        except shapely.errors.GEOSException:
            d["geometry"] = sg.Point(coords)

        return pd.Series(d)

    g = (
        shapes.sort_values(["shape_id", "shape_pt_sequence"])
        .groupby("shape_id", sort=False)
        .apply(my_agg, include_groups=False)
        .reset_index()
        .pipe(gpd.GeoDataFrame)
        .set_crs(cs.WGS84)
    )

    if use_utm:
        g = g.to_crs(g.estimate_utm_crs())

    return g


def ungeometrize_shapes(shapes_g: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    The inverse of :func:`geometrize_shapes`.

    If ``shapes_g`` is in UTM coordinates (has a UTM CRS property),
    then convert those UTM coordinates back to WGS84 coordinates,
    which is the standard for a GTFS shapes table.
    """
    shapes_g = shapes_g.to_crs(cs.WGS84)

    F = []
    for index, row in shapes_g.iterrows():
        F.extend(
            [
                [row["shape_id"], i, x, y]
                for i, (x, y) in enumerate(row["geometry"].coords)
            ]
        )

    return pd.DataFrame(
        F,
        columns=[
            "shape_id",
            "shape_pt_sequence",
            "shape_pt_lon",
            "shape_pt_lat",
        ],
    ).astype({"shape_id": "string"})


def get_shapes(
    feed: "Feed", *, as_gdf: bool = False, use_utm: bool = False
) -> gpd.DataFrame | None:
    """
    Get the shapes DataFrame for the given feed, which could be ``None``.
    If ``as_gdf``, then return it as GeoDataFrame with a 'geometry' column
    of linestrings and no 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat',
    'shape_dist_traveled' columns.
    The GeoDataFrame will have a UTM CRS if ``use_utm``; otherwise it will have a
    WGS84 CRS.
    """
    f = feed.shapes
    if f is not None and as_gdf:
        f = geometrize_shapes(f, use_utm=use_utm)
    return f


def build_geometry_by_shape(
    feed: "Feed", shape_ids: Iterable[str] | None = None, *, use_utm: bool = False
) -> dict:
    """
    Return a dictionary of the form
    <shape ID> -> <Shapely LineString representing shape>.
    If the Feed has no shapes, then return the empty dictionary.
    If ``use_utm``, then use local UTM coordinates; otherwise, use WGS84 coordinates.
    """
    if feed.shapes is None:
        return dict()

    g = get_shapes(feed, as_gdf=True, use_utm=use_utm)
    if shape_ids is not None:
        g = g.loc[lambda x: x["shape_id"].isin(shape_ids)]
    return dict(g[["shape_id", "geometry"]].values)


def shapes_to_geojson(feed: "Feed", shape_ids: Iterable[str] | None = None) -> dict:
    """
    Return a GeoJSON FeatureCollection of LineString features
    representing ``feed.shapes``.
    If the Feed has no shapes, then the features will be an empty list.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If an iterable of shape IDs is given, then subset to those shapes.
    If the subset is empty, then return a FeatureCollection with an empty list of
    features.
    """
    g = get_shapes(feed, as_gdf=True)
    if shape_ids is not None:
        g = g.loc[lambda x: x["shape_id"].isin(shape_ids)]
    if g is None or g.empty:
        result = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        result = json.loads(g.to_json(drop_id=True))
    return result


def get_shapes_intersecting_geometry(
    feed: "Feed",
    geometry: sg.base.BaseGeometry,
    shapes_g: gpd.GeoDataFrame | None = None,
    *,
    as_gdf: bool = False,
) -> pd.DataFrame | None:
    """
    If the Feed has no shapes, then return None.
    Otherwise, return the subset of ``feed.shapes`` that contains all shapes that
    intersect the given Shapely WGS84 geometry, e.g. a Polygon or LineString.

    If ``as_gdf``, then return the shapes as a GeoDataFrame.
    Specifying ``shapes_g`` will skip the first step of the
    algorithm, namely, geometrizing ``feed.shapes``.
    """
    if feed.shapes is None:
        return None

    if shapes_g is not None:
        g = shapes_g.copy()
    else:
        g = get_shapes(feed, as_gdf=True)

    cols = g.columns
    g["hit"] = g["geometry"].intersects(geometry)
    g = g.loc[lambda x: x["hit"]].filter(cols)

    if as_gdf:
        result = g
    else:
        result = ungeometrize_shapes(g)

    return result


def split_simple_0(ls: sg.LineString) -> list[sg.LineString]:
    """
    Split the given LineString into simple sub-LineStrings by greedily building
    the segments from the LineString points and binary search,
    checking for simplicity at every step.
    """
    coords = ls.coords
    segments = []
    n = len(coords)
    i = 0
    while i < n:
        # If only one coordinate remains, break  to
        # avoids making a degenerate LineString
        if i == n - 1:
            break
        # Start a binary search with at least two points
        lo, hi = i + 1, n - 1
        best = i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = sg.LineString(coords[i : mid + 1])
            if candidate.is_simple:
                best = mid  # candidate is simple; try extending further.
                lo = mid + 1
            else:
                hi = mid - 1
        segments.append(sg.LineString(coords[i : best + 1]))
        if best == n - 1:
            break
        # Start next segment from current best segment
        i = best
    return segments


def split_simple(shapes_g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Given a GeoDataFrame of GTFS shapes of the form output by :func:`geometrize_shapes`
    with possibly non-WGS84 coordinates,
    split each non-simple LineString into large simple (non-self-intersecting)
    sub-LineStrings, and leave the simple LineStrings as is.

    Return a GeoDataFrame in the coordinates of ``shapes_g`` with the columns

    - ``'shape_id'``: GTFS shape ID for a LineString L
    - ``'subshape_id'``: a unique identifier of a simple sub-LineString S of L
    - ``'subshape_sequence'``: integer; indicates the order of S when joining up
      all simple sub-LineStrings to form L
    - ``'subshape_length_m'``: the length of S in meters
    - ``'cum_length_m'``: the length S plus the lengths of sub-LineStrings of L
      that come before S; in meters
    - ``'geometry'``: LineString geometry corresponding to S

    Within each 'shape_id' group, the subshapes will be sorted increasingly by
    'subshape_sequence'.

    Notes
    -----
    - Simplicity checks and splitting are done in local UTM coordinates.
      Converting back to original coordinates can introduce rounding errors and
      non-simplicities.
      So test this function with a ``shapes_g`` in local UTM coordinates.
    - By construction, for each given LineString L with simple sub-LineStrings
      S_i, we have the inequality

        sum over i of length(S_i) <= length(L),

      where the lengths are expressed in meters.

    """
    orig_crs = shapes_g.crs
    utm_crs = shapes_g.estimate_utm_crs()
    g = shapes_g.to_crs(utm_crs).assign(is_simple=lambda x: x.is_simple)
    final_cols = [
        "subshape_id",
        "subshape_sequence",
        "shape_id",
        "subshape_length_m",
        "cum_length_m",
        "geometry",
    ]

    # Simple shapes don't need splitting
    g0 = (
        g.loc[lambda x: x["is_simple"]]
        .assign(
            subshape_id=lambda x: x["shape_id"].astype(str) + "-0",
            subshape_sequence=0,
            subshape_length_m=lambda x: x.length,
            cum_length_m=lambda x: x["subshape_length_m"],
        )
        .filter(final_cols)
    )

    # Split the non-simple shapes
    def my_split(group):
        geom = group["geometry"].iat[0]
        parts = split_simple_0(geom)
        return pd.DataFrame({"geometry": parts})

    g1 = (
        g.loc[lambda x: ~x["is_simple"]]
        .groupby("shape_id", sort=False)
        .apply(my_split, include_groups=False)
        .reset_index()
        .rename(columns={"level_1": "subshape_sequence"})
        .assign(
            subshape_id=lambda x: x["shape_id"].str.cat(
                x["subshape_sequence"].astype(str), sep="-"
            )
        )
        .pipe(gpd.GeoDataFrame)
        .set_crs(utm_crs)
        .assign(
            subshape_length_m=lambda x: x.length,
            cum_length_m=lambda x: x.groupby("shape_id")["subshape_length_m"].cumsum(),
        )
        .filter(final_cols)
    )
    return (
        pd.concat([g0, g1])
        .sort_values(["shape_id", "subshape_sequence"], ignore_index=True)
        .to_crs(orig_crs)
    )
