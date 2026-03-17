Introduction
=============
GTFS Kit is an open-source Python library for analyzing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It uses Pandas and GeoPandas to do the heavy lifting.
You can find and contribute to GTFS Kit's source code in its project repository `on Github <https://github.com/mrcagney/gtfs_kit>`_.

The functions/methods of GTFS Kit assume a valid GTFS feed but offer no inbuilt validation, because GTFS validation is complex and already solved by dedicated libraries.
So unless you know what you're doing, use the `Canonical GTFS Validator <https://gtfs-validator.mobilitydata.org/>`_ before you analyze a feed with GTFS Kit.


Authors
=========
- Alex Raichev, 2019-09


Installation
=============
Install it from PyPI with UV, say, via ``uv add gtfs_kit``.


Examples
========
See the `Marimo notebook output on Github <https://github.com/mrcagney/gtfs_kit/blob/master/notebooks/examples.ipynb>`_.

Conventions
============
- In conformance with GTFS, dates are encoded as YYYYMMDD date strings, and times are encoded as HH:MM:SS time strings with the possibility that HH > 24. **Watch out** for that possibility, because it has counterintuitive consequences; see e.g. :func:`.trips.is_active_trip`, which is used in :func:`.routes.compute_route_stats`,  :func:`.stops.compute_stop_stats`, and :func:`.miscellany.compute_network_stats`.
- 'DataFrame' and 'Series' refer to Pandas DataFrame and Series objects,
  respectively
