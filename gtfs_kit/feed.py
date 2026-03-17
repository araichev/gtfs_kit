"""
This module defines a Feed class to represent GTFS feeds.

The Feed class has one instance attribute for each GTFS table and exposes
many convenience methods for analysis and transformation.
Most method implementations live as functions in themed sibling modules such as
`routes.py`, `stops.py`, and `trips.py`.
A select set of them is then attached to `Feed`,
so the implementation remains modular while the public Feed API stays explicit.
"""

import pathlib as pl
import shutil
import tempfile
import zipfile
from functools import wraps

import pandas as pd
import requests


from . import constants as cs
from . import calendar as cl
from . import cleaners as cn
from . import helpers as hp
from . import miscellany as ms
from . import routes as rt
from . import shapes as shp
from . import stop_times as st
from . import stops as sp
from . import trips as tr


class Feed:
    """
    An instance of this class represents a not-necessarily-valid GTFS feed,
    where GTFS tables are stored as DataFrames.
    Beware that the stop times DataFrame can be big (several gigabytes),
    so make sure you have enough memory to handle it.

    Primary instance attributes:

    - ``dist_units``: a string in :const:`.constants.DIST_UNITS`;
      specifies the distance units of the `shape_dist_traveled` column values,
      if present; also effects whether to display trip and route stats in
      metric or imperial units
    - ``agency``
    - ``stops``
    - ``routes``
    - ``trips``
    - ``stop_times``
    - ``calendar``
    - ``calendar_dates``
    - ``fare_attributes``
    - ``fare_rules``
    - ``shapes``
    - ``frequencies``
    - ``transfers``
    - ``feed_info``
    - ``attributions``

    There are also a few secondary instance attributes that are derived
    from the primary attributes and are automatically updated when the
    primary attributes change.
    However, for this update to work, you must update the primary
    attributes like this (good)::

        feed.trips['route_short_name'] = 'bingo'
        feed.trips = feed.trips

    and **not** like this (bad)::

        feed.trips['route_short_name'] = 'bingo'

    The first way ensures that the altered trips DataFrame is saved as
    the new ``trips`` attribute, but the second way does not.

    """

    def __init__(
        self,
        dist_units: str,
        agency: pd.DataFrame | None = None,
        stops: pd.DataFrame | None = None,
        routes: pd.DataFrame | None = None,
        trips: pd.DataFrame | None = None,
        stop_times: pd.DataFrame | None = None,
        calendar: pd.DataFrame | None = None,
        calendar_dates: pd.DataFrame | None = None,
        fare_attributes: pd.DataFrame | None = None,
        fare_rules: pd.DataFrame | None = None,
        shapes: pd.DataFrame | None = None,
        frequencies: pd.DataFrame | None = None,
        transfers: pd.DataFrame | None = None,
        feed_info: pd.DataFrame | None = None,
        attributions: pd.DataFrame | None = None,
    ):
        """
        Assume that every non-None input is a DataFrame,
        except for ``dist_units`` which should be a string in
        :const:`.constants.DIST_UNITS`.

        No other format checking is performed.
        In particular, a Feed instance need not represent a valid GTFS
        feed.
        """
        # Set primary attributes from inputs.
        # The @property magic below will then
        # validate some and set some derived attributes
        for prop, val in locals().items():
            if prop in cs.FEED_ATTRS:
                setattr(self, prop, val)

    @property
    def dist_units(self):
        """
        The distance units of the Feed.
        """
        return self._dist_units

    @dist_units.setter
    def dist_units(self, val):
        if val not in cs.DIST_UNITS:
            raise ValueError(
                f"Distance units are required and must lie in {cs.DIST_UNITS}"
            )
        else:
            self._dist_units = val

    def __str__(self):
        """
        Print the first five rows of each GTFS table.
        """
        d = {}
        for table in cs.DTYPES:
            try:
                d[table] = getattr(self, table).head(5)
            except Exception:
                d[table] = None
        d["dist_units"] = self.dist_units

        return "\n".join([f"* {k} --------------------\n\t{v}" for k, v in d.items()])

    def __eq__(self, other):
        """
        Define two feeds be equal if and only if their
        :const:`.constants.FEED_ATTRS` attributes are equal,
        or almost equal in the case of DataFrames
        (but not groupby DataFrames).
        Almost equality is checked via :func:`.helpers.almost_equal`,
        which   canonically sorts DataFrame rows and columns.
        """
        # Return False if failures
        for key in cs.FEED_ATTRS:
            x = getattr(self, key)
            y = getattr(other, key)
            # DataFrame case
            if isinstance(x, pd.DataFrame):
                if not isinstance(y, pd.DataFrame) or not hp.almost_equal(x, y):
                    return False
            # Other case
            else:
                if x != y:
                    return False
        # No failures
        return True

    def copy(self) -> "Feed":
        """
        Return a copy of this feed, that is, a feed with all the same
        attributes.
        """
        other = Feed(dist_units=self.dist_units)
        for key in set(cs.FEED_ATTRS) - set(["dist_units"]):
            value = getattr(self, key)
            if isinstance(value, pd.DataFrame):
                # Pandas copy DataFrame
                value = value.copy()
            setattr(other, key, value)

        return other

    def to_file(self, path: pl.Path, ndigits: int | None = None) -> None:
        """
        Write this Feed to the given path.
        If the path ends in '.zip', then write the feed as a zip archive.
        Otherwise assume the path is a directory, and write the feed as a
        collection of CSV files to that directory, creating the directory
        if it does not exist.
        Round all decimals to ``ndigits`` decimal places, if given.
        All distances will be the distance units ``feed.dist_units``.
        By the way, 6 decimal degrees of latitude and longitude is enough to locate
        an individual cat.
        """
        path = pl.Path(path)

        if path.suffix == ".zip":
            # Write to temporary directory before zipping
            zipped = True
            tmp_dir = tempfile.TemporaryDirectory()
            new_path = pl.Path(tmp_dir.name)
        else:
            zipped = False
            if not path.exists():
                path.mkdir()
            new_path = path

        for table in cs.DTYPES:
            f = getattr(self, table)
            if f is not None:
                p = new_path / (table + ".txt")
                if ndigits is not None:
                    f.to_csv(p, index=False, float_format=f"%.{ndigits}f")
                else:
                    f.to_csv(p, index=False)

        # Zip directory
        if zipped:
            basename = str(path.parent / path.stem)
            shutil.make_archive(basename, format="zip", root_dir=tmp_dir.name)
            tmp_dir.cleanup()


# Attach select functions as methods to Feed
def _make_feed_method(func):
    @wraps(func)
    def method(self, *args, **kwargs):
        return func(self, *args, **kwargs)

    return method


_FEED_METHODS = {
    # calendar
    "get_dates": cl.get_dates,
    "get_first_week": cl.get_first_week,
    "get_week": cl.get_week,
    "subset_dates": cl.subset_dates,
    # cleaners
    "aggregate_routes": cn.aggregate_routes,
    "aggregate_stops": cn.aggregate_stops,
    "clean": cn.clean,
    "clean_ids": cn.clean_ids,
    "clean_route_short_names": cn.clean_route_short_names,
    "clean_times": cn.clean_times,
    "drop_invalid_columns": cn.drop_invalid_columns,
    "drop_zombies": cn.drop_zombies,
    "extend_id": cn.extend_id,
    # miscellany
    "assess_quality": ms.assess_quality,
    "compute_bounds": ms.compute_bounds,
    "compute_centroid": ms.compute_centroid,
    "compute_convex_hull": ms.compute_convex_hull,
    "compute_network_stats": ms.compute_network_stats,
    "compute_network_time_series": ms.compute_network_time_series,
    "compute_screen_line_counts": ms.compute_screen_line_counts,
    "convert_dist": ms.convert_dist,
    "create_shapes": ms.create_shapes,
    "describe": ms.describe,
    "list_fields": ms.list_fields,
    "restrict_to_agencies": ms.restrict_to_agencies,
    "restrict_to_area": ms.restrict_to_area,
    "restrict_to_dates": ms.restrict_to_dates,
    "restrict_to_routes": ms.restrict_to_routes,
    "restrict_to_trips": ms.restrict_to_trips,
    # routes
    "build_route_timetable": rt.build_route_timetable,
    "compute_route_stats": rt.compute_route_stats,
    "compute_route_time_series": rt.compute_route_time_series,
    "get_routes": rt.get_routes,
    "map_routes": rt.map_routes,
    "routes_to_geojson": rt.routes_to_geojson,
    # shapes
    "append_dist_to_shapes": shp.append_dist_to_shapes,
    "build_geometry_by_shape": shp.build_geometry_by_shape,
    "geometrize_shapes": shp.geometrize_shapes,
    "get_shapes": shp.get_shapes,
    "get_shapes_intersecting_geometry": shp.get_shapes_intersecting_geometry,
    "shapes_to_geojson": shp.shapes_to_geojson,
    "split_simple": shp.split_simple,
    # stop_times
    "append_dist_to_stop_times": st.append_dist_to_stop_times,
    "get_start_and_end_times": st.get_start_and_end_times,
    "get_stop_times": st.get_stop_times,
    "stop_times_to_geojson": st.stop_times_to_geojson,
    # stops
    "build_geometry_by_stop": sp.build_geometry_by_stop,
    "build_stop_timetable": sp.build_stop_timetable,
    "compute_stop_activity": sp.compute_stop_activity,
    "compute_stop_stats": sp.compute_stop_stats,
    "compute_stop_time_series": sp.compute_stop_time_series,
    "geometrize_stops": sp.geometrize_stops,
    "get_stops": sp.get_stops,
    "get_stops_in_area": sp.get_stops_in_area,
    "map_stops": sp.map_stops,
    "stops_to_geojson": sp.stops_to_geojson,
    "ungeometrize_stops": sp.ungeometrize_stops,
    # trips
    "compute_busiest_date": tr.compute_busiest_date,
    "compute_trip_activity": tr.compute_trip_activity,
    "compute_trip_stats": tr.compute_trip_stats,
    "get_active_services": tr.get_active_services,
    "get_trips": tr.get_trips,
    "locate_trips": tr.locate_trips,
    "map_trips": tr.map_trips,
    "name_stop_patterns": tr.name_stop_patterns,
    "trips_to_geojson": tr.trips_to_geojson,
}
for name, func in _FEED_METHODS.items():
    setattr(Feed, name, _make_feed_method(func))

# Clean up namespace
del name, func


# -------------------------------------
# Functions about input and output
# -------------------------------------
def list_feed(path: pl.Path) -> pd.DataFrame:
    """
    Given a path (string or Path object) to a GTFS zip file or
    directory, record the file names and file sizes of the contents,
    and return the result in a DataFrame with the columns:

    - ``'file_name'``
    - ``'file_size'``
    """
    path = pl.Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    # Collect rows of DataFrame
    rows = []
    if path.is_file():
        # Zip file
        with zipfile.ZipFile(str(path)) as src:
            for x in src.infolist():
                if x.filename == "./":
                    continue
                d = {}
                d["file_name"] = x.filename
                d["file_size"] = x.file_size
                rows.append(d)
    else:
        # Directory
        for x in path.iterdir():
            d = {}
            d["file_name"] = x.name
            d["file_size"] = x.stat().st_size
            rows.append(d)

    return pd.DataFrame(rows)


def _read_feed_from_path(path: pl.Path, dist_units: str) -> "Feed":
    """
    Helper function for :func:`read_feed`.
    Create a Feed instance from the given path and given distance units.
    The path should be a directory containing GTFS text files or a
    zip file that unzips as a collection of GTFS text files
    (and not as a directory containing GTFS text files).
    The distance units given must lie in :const:`constants.dist_units`

    Notes:

    - Ignore non-GTFS files in the feed
    - Automatically strip whitespace from the column names in GTFS files

    """
    path = pl.Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    # Read files into feed dictionary of DataFrames
    feed_dict = {table: None for table in cs.DTYPES}
    csv_options = {
        "na_values": ["", " ", "nan", "NaN", "null"],  # Add space to na_values
        "keep_default_na": True,
        "dtype_backend": "numpy_nullable",
        # utf-8-sig gets rid of the byte order mark (BOM);
        # see http://stackoverflow.com/questions/17912307/u-ufeff-in-python-string
        "encoding": "utf-8-sig",
    }

    if path.is_file():
        with zipfile.ZipFile(path, "r") as zf:
            for file_info in zf.infolist():
                table = pl.Path(file_info.filename).stem
                # Skip empty files, irrelevant files, and files with no data
                if (
                    not file_info.is_dir()
                    and file_info.file_size
                    and file_info.filename.endswith(".txt")
                    and table in feed_dict
                ):
                    with zf.open(file_info.filename) as f:
                        df = pd.read_csv(f, dtype=cs.DTYPES[table], **csv_options)
                        if not df.empty:
                            feed_dict[table] = cn.clean_column_names(df)
    else:
        for p in path.iterdir():
            table = p.stem
            # Skip empty files, irrelevant files, and files with no data
            if (
                p.is_file()
                and p.stat().st_size
                and p.suffix == ".txt"
                and table in feed_dict
            ):
                df = pd.read_csv(p, dtype=cs.DTYPES[table], **csv_options)
                if not df.empty:
                    feed_dict[table] = cn.clean_column_names(df)

    feed_dict["dist_units"] = dist_units

    # Create feed
    return Feed(**feed_dict)


def _read_feed_from_url(url: str, dist_units: str) -> "Feed":
    """
    Helper function for :func:`read_feed`.

    Create a Feed instance from the given URL and given distance units.
    Notes:
    - Ignore non-GTFS files in the feed
    - Automatically strip whitespace from the column names in GTFS files
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pl.Path(tmp_dir) / "feed.zip"

        with requests.get(url, stream=True, timeout=(10, 60)) as r:
            r.raise_for_status()

            content_type = (r.headers.get("Content-Type") or "").lower()

            # Content-Type is only a heuristic: accept common zip-ish values,
            # tolerate a missing header, but reject obvious non-zip responses.
            if content_type and not any(
                token in content_type
                for token in (
                    "application/zip",
                    "application/x-zip-compressed",
                    "application/octet-stream",
                    "application/binary",
                    "multipart/x-zip",
                    "application/x-download",
                )
            ):
                raise ValueError(f"URL returned a non-zip content type: {content_type!r}")

            with tmp_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Final authority: verify the downloaded payload is actually a zip file.
        if not zipfile.is_zipfile(tmp_path):
            raise ValueError("Downloaded content is not a valid zip archive")

        return _read_feed_from_path(tmp_path, dist_units=dist_units)


def read_feed(path_or_url: pl.Path | str, dist_units: str) -> "Feed":
    """
    Create a Feed instance from the given path or URL and given distance units.

    Notes:
    - Ignore non-GTFS files in the feed
    - Automatically strip whitespace from the column names in GTFS files
    """
    try:
        path = pl.Path(path_or_url)
        if path.exists():
            return _read_feed_from_path(path, dist_units=dist_units)
    except OSError:
        pass

    try:
        return _read_feed_from_url(str(path_or_url), dist_units=dist_units)
    except (requests.RequestException, ValueError, zipfile.BadZipFile) as e:
        raise ValueError(
            f"Path does not exist or URL could not be read as a GTFS zip: {e}"
        ) from e
