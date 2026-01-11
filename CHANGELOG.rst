Changelog
=========

12.0.2, 2026-01-12
------------------
- Sped up reading GTFS feeds from file and a few other functions, thanks to user ``abdelq`` and `pull request 37 <https://github.com/araichev/gtfs_kit/pull/37>`_.
- Handled the case of missing ``parent_station`` column in ``miscellany.restrict_to_trips``, thereby remedying `issue 39<https://github.com/araichev/gtfs_kit/issues/39>`_.
- Updated docs URL in README.
- Updated dependencies.

12.0.1, 2025-11-06
------------------
- Fixed and improved the docs.

12.0.0, 2025-10-22
------------------
- Breaking change: Refactored and sped up functions ``shapes.split_simple`` and ``miscellany.compute_screen_line_counts`` to no longer segmentize linestrings, then removed the `segmentize` keyword argument from both functions.
- Added ``route_short_names`` keyword argument to ``routes.routes_to_geojson``.
- Silently dropped invalid stops, trips, and routes from ``stops.stops_to_geojson``, ``trips.trips_to_geojson``, and ``routes.routes_to_geojson`` instead of raising ``ValueError``s.

11.0.1, 2025-10-10
------------------
- Updated ``routes.get_routes(as_gdf=True)`` to ignore ``None`` geometries, fixing a bug spotted by Github user khamaileon.

11.0.0, 2025-09-26
------------------
- Breaking change: simplified ``constants.py`` and changed all ``Int8`` entries to ``Int32`` to handle some extended route types etc.
- Breaking change: created separate inverse functions for time helpers, e.g. ``seconds_to_timestr`` for ``timestr_to_seconds``.
- Breaking change: renamed ``compute_feed_stats`` and ``compute_feed_time_series`` to ``compute_network_stats`` and ``compute_network_time_series``, respectively.
- Breaking change: simplified stop/route/network time series to long tables with no hierarchical columns.
- Breaking change: simplified signatures of ``compute_route_stats``, ``compute_route_time_series``, ``compute_network_stats``, and ``compute_network_time_series`` to make trip stats optional.

10.3.1, 2025-06-04
------------------
- Bugfixed ``routes.get_routes``, which was deleting shapes shared across routes.
- Added to ``notebooks/examples.py`` an example of plotting routes with GeoPandas's ``explore`` method.

10.3.0, 2025-04-30
------------------
- In ``cleaners.drop_zombies``, additionally dropped agencies without routes.
- Fixed ``routes.get_routes`` to handle the case when at least one of a route's shapes is a point.
- Turned the example notebook into a Marimo notebook.

10.2.2, 2025-03-13
------------------
- Bugfixed an edge case in the function ``shapes.split_simple``.

10.2.1, 2025-03-12
------------------
- Added a forgotten ``drop_duplicates('shape_id')`` line to the function ``miscellany.compute_screen_line_counts``, which speeds things up on large sets of screen lines.

10.2.0, 2025-03-12
------------------
- Improved ``miscellany.compute_screen_line_counts`` to now properly handle trips with non-simple shapes.
- Added ``shapes.split_simple`` to help with the new screen line computation, but users might find it useful for other things too.

10.1.1, 2025-01-21
------------------
- Bugfixed ``trips.compute_trip_activity`` which cleverly and wrongly cast values to integers.
- Improved ``trips.get_active_services`` to handle feeds with only one of ``calendar.txt`` and ``calendar_dates.txt``.

10.1.0, 2025-01-09
------------------
- Handled null values better thanks to `pull request 22 <https://github.com/mrcagney/gtfs_kit/pull/22>`_.
- Updated ``stop_times.append_dist_to_stop_times`` to handle trips missing shapes by setting their distances to NaN.
- Updated ``miscellany.restrict_to_trips`` to also include the parent stations of the trip stops.
- Forgot to import ``miscellany.restrict_to_trips`` in ``feed`` in last release. Fixed that.

10.0.0, 2024-12-20
------------------
- Added ``miscellany.restrict_to_trips`` and used it as a helper function to simplify the other restriction functions.
- Breaking change: Removed the validation module ``validators.py`` to avoid duplicating the work of what is now `the canonical feed validator <https://github.com/MobilityData/gtfs-validator>`_ (written in Java).
- Breaking change: Changed ``feed.write`` to ``feed.to_file`` and stopped default rounding.
- Breaking change: Changed ``miscellany.summarize`` to ``miscellany.list_fields`` and stopped default rounding.

9.0.0, 2024-12-19
-----------------
- Breaking change: Replaced ``trips.is_active_trip`` with ``trips.get_active_services`` and removed the derived feed attributes ``trips_i``, ``calendar_i``, and ``calendar_dates_i`` as no longer necessary and overly complex.

8.1.4, 2024-12-19
-----------------
- Added ``restrict_to_agencies`` to ``feed`` module local imports. Whoops!

8.1.3, 2024-12-19
-----------------
- Added ``miscellany.restrict_to_agencies``, thanks to Github user `diegoperezalvarez`.

8.1.2, 2024-12-16
-----------------
- Fixed sorting in ``stops.build_timetable`` and ``routes.build_timetable``.
- Improved data types for CSV reads.
- Fixed Pandas groupby deprecation warnings.
- Ignored Shapely runtime warnings in tests.

8.1.1, 2024-10-31
-----------------
- Bugfixed ``shapes.geometrize_shapes`` to handle shapes comprising a single point.

8.1.0, 2024-10-09
-----------------
- Added function ``trips.name_stop_patterns``, then used it to append column ``stop_pattern_name`` to the output of ``trips.compute_trip_stats`` and to append column ``num_stop_patterns`` to the output of ``routes.compute_route_stats``.

8.0.0, 2024-10-08
-----------------
- Breaking change: removed the UTM library, deleted ``helpers.get_utm_crs``, and used the GeoPandas version of the function instead.
- Changed ``routes.map_routes`` to accept a list of route short names, instead of or in addition to a list of route IDs.

7.0.0, 2024-09-30
-----------------
- Switched from Poetry to UV for project management.
- Breaking change: removed ``geometrize_stops`` function and moved its functionality into ``get_stops``. Did a similar thing for ``get_shapes``, ``get_trips``, and ``get_routes``.

6.1.1, 2024-08-19
-----------------
- Changed grouped DataFrame ``feed._calendar_dates_g`` to indexed DataFrame ``feed._calendar_dates_i`` for consistency with ``feed._calendar_i`` and slight speedup in fucttion ``trips.is_active_trip``.
- Updated dependencies and dropped Python 3.8 support.
- Addressed some Pandas deprecation warnings.

6.1.0, 2024-02-02
-----------------
- Added ``cleaners.extend_id`` function in response to `Pull Request 7 <https://github.com/mrcagney/gtfs_kit/pull/7>`_.

6.0.1, 2024-01-30
-----------------
- Fixed a new GeoPandas 'set geometry' error.

6.0.0, 2023-10-03
-----------------
- Changed keywords in ``map_trips()`` and ``map_routes()`` because i keep remembering them wrong.

5.2.8, 2023-07-21
-----------------
- Bugfixed the ``check_attributions()`` validator.
- Changed ``compute_route_stats_0`` to ignore trips of zero duration, thereby addressing a different aspect of `Issue 2 <https://github.com/mrcagney/gtfs_kit/issues/2>`_.
- Updated dependencies.

5.2.7, 2023-06-06
-----------------
- Bugfixed ``get_peak_indices``, addressing `Issue 2 <https://github.com/mrcagney/gtfs_kit/issues/2>`_.

5.2.6, 2023-05-30
-----------------
- Bugfixed ``geometrize_routes``, addressing `Issue 1 <https://github.com/mrcagney/gtfs_kit/issues/1>`_.
- Removed star imports from tests.
- Updated dependencies.

5.2.5, 2023-04-26
-----------------
- Updated dependencies.
- Updated ``compute_screen_line_counts`` for Shapely >=2.

5.2.4, 2023-03-22
-----------------
- Updated dependencies and pre-commit hooks.
- Added a Github Action for testing.

5.2.3, 2022-06-28
-----------------
- Upgraded to Python 3.10 and updated dependencies.

5.2.2, 2022-04-27
-----------------
- Fixed ``transfer_type`` range in ``validators.py``.
  Was 0,...,4 but should have been 0,..,3.

5.2.1, 2022-04-12
-----------------
- Updated dependencies and removed version caps.
- Updated README.

5.2.0, 2022-01-17
-----------------
- Added support for ``attributions.txt``.
- Fixed ``aggregate_stops()`` docstring.

5.1.4, 2021-05-19
-----------------
- Bugfixed ``geometrize_routes(use_utm=True)`` to actually use UTM coordinates.

5.1.3, 2021-05-19
-----------------
- Bugfixed distance units in trip stats when shape_dist_traveled is present.

5.1.2, 2021-05-17
-----------------
- Changed distance units in trip stats, route stats, and feed stats to kilometers if the feed's distance units are metric and to miles otherwise.
- Added stop time information to stops when mapping trips with stops.

5.1.1, 2021-04-30
-----------------
- Handled fare rules in ``aggregate_routes()`` and dropped mistaken transfers code block therein.

5.1.0, 2021-04-29
-----------------
- Added support for Python 3.9 and dropped support for Python 3.6.

5.0.2, 2020-10-16
-----------------
- Specified in more detail the Rtree dependency.

5.0.1, 2020-10-08
-----------------
- Bugfix: properly set the ``use_utm`` flag in ``build_geometry_by_shape()`` and ``build_geometry_by_stop()``.

5.0.0, 2020-06-16
-----------------
- Breaking change: refactored ``get_stops_in_polygon()`` to ``get_stops_in_area()``, which accepts a GeoDataFrame.
- Breaking change: refactored ``restrict_to_polygon()`` to ``restrict_to_area()``, which accepts a GeoDataFrame.
- Breaking changes: refactored ``compute_center()`` to ``compute_centroid()``.
- Updated ``get_utm_crs()`` to differentiate between northern and southern hemispheres.
- Added more defensive copying after subsetting some DataFrames.
- Fixed calendar_dates table in ``restrict_to_dates()``.
- Added ``compute_convex_hull()`` to Feed methods. Forgot about that function.
- Switched from using route IDs to using route short names for layer names in ``map_routes()``.

4.0.2, 2020-05-07
-----------------
- Fixed a CRS deprecation warning as requested in `Pull Request 5 <https://github.com/mrcagney/gtfs_kit/pull/5>`_.
- Changed ``get_utm_crs()`` to output an EPSG CRS string, e.g. "EPSG:32655", instead of a PROJ4 definition string. Did this under the recommendation of the `GeoPandas docs <https://geopandas.org/projections.html#manually-specifying-the-crs>`_.
- Fixed CRS mismatch warning in ``compute_screen_line_counts()``.
- Updated dependencies and included Python 3.8 support.

4.0.1, 2020-04-24
-----------------
- Bugfix: got ``read_feed()`` working on Windows thanks to `Pull Request 4 <https://github.com/mrcagney/gtfs_kit/pull/4>`_.

4.0.0, 2020-03-06
-----------------
- Breaking changes: renamed ``list_gtfs()`` to ``list_feed()``, ``read_gtfs()`` to ``read_feed()``, and ``write_gtfs()`` to ``write()`` and made it a Feed method.
- Made ``read_feed()`` accept URLs as requested in `Pull Request 3 <https://github.com/mrcagney/gtfs_kit/pull/3>`_.

3.0.1, 2020-01-16
-----------------
- Optimized function ``geometrize_routes()`` by ignoring duplicate shapes.

3.0.0, 2020-01-10
-----------------
- Breaking change: improved function ``compute_screen_line_counts()`` to handle multiple screen lines at once.
- Added helper function ``make_ids()``.

2.2.1, 2019-11-07
-----------------
- Bugfix: updated function ``map_trips()`` to heed the ``include_arrows`` parameter.

2.2.0, 2019-10-31
-----------------
- Modularized some by added the functions ``build_aggregate_routes_dict()`` and ``build_aggregate_stops_dict()``.

2.1.0, 2019-10-10
-----------------
- Bugfix: updated ``aggregate_stops()`` to handle parent stations.
- Added optional direction arrows to ``map_trips()``.

2.0.0, 2019-10-04
-----------------
- Improved the fallback algorithm in ``append_dist_to_stop_times()``. Changed the function signature, so this is a major change, hence the major version bump.
- Removed optional direction arrows in ``map_trips()``, because the PolyLineTextPath Folium plugin needed for that seems to be broken.

1.1.1, 2019-09-25
-----------------
- Bugfix: imported ``aggregate_stops()`` as a Feed method.

1.1.0, 2019-09-25
-----------------
- Added ``aggregate_stops()`` function.
- Added optional direction arrows in ``map_trips()``.

1.0.2, 2019-09-20
-----------------
- Bugfix: Fixed CRS in ``geometrize_trips()`` and ``geometrize_routes()`` when ``use_utm=True``.

1.0.1, 2019-09-20
-----------------
- Bugfixed: Fixed occasional indexing error in ``geometrize_stops()`` and ``geometrize_shapes()`` when ``use_utm=True``.

1.0.0, 2019-09-18
-----------------
- First release based on prior work.
