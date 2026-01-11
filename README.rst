GTFS Kit
********
.. image:: https://github.com/mrcagney/gtfs_kit/actions/workflows/test.yml/badge.svg

GTFS Kit is a Python 3.10+ library for analyzing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It uses Pandas and GeoPandas to do the heavy lifting.

The functions/methods of GTFS Kit assume a valid GTFS feed but offer no inbuilt validation, because GTFS validation is complex and already solved by dedicated libraries.
So unless you know what you're doing, use the `Canonical GTFS Validator <https://gtfs-validator.mobilitydata.org/>`_ before you analyze a feed with GTFS Kit.


Installation
=============
Install it from PyPI with UV, say, via ``uv add gtfs_kit``.


Examples
========
In the Jupyter notebook ``notebooks/examples.ipynb``, which is a Github-displayable export of the Marimo notebook ``notebooks/examples.py``.


Authors
=========
- Alex Raichev (2019-09), maintainer


Documentation
=============
The documentation is built via Sphinx from the source code in the ``docs`` directory then published to Github Pages at `araichev.github.io/gtfs_kit_docs <https://araichev.github.io/gtfs_kit_docs>`_.


Notes
=====
- This project's development status is Alpha.
  I use GTFS Kit at my job and change it breakingly to suit my needs.
- This project uses semantic versioning.
- I aim for GTFS Kit to handle `the current GTFS <https://developers.google.com/transit/gtfs/reference>`_.
  In particular, i avoid handling `GTFS extensions <https://developers.google.com/transit/gtfs/reference/gtfs-extensions>`_.
  That is the most reasonable scope boundary i can draw at present, given this project's tiny budget.
  If you would like to fund this project to expand its scope, please email me.
- Thanks to `MRCagney <http://www.mrcagney.com/>`_ for periodically donating to this project.
- Constructive feedback and contributions are welcome.
  Please issue pull requests from a feature branch into the ``develop`` branch and include tests.
- GTFS time is measured relative to noon minus 12 hours, which can mess things up when crossing into daylight savings time.
  I don't think this issue causes any bugs in GTFS Kit, but you and i have been warned.
  Thanks to user Github user ``derhuerst`` for bringing this to my attention in `closed Issue 8 <https://github.com/mrcagney/gtfs_kit/issues/8#issue-1063633457>`_.
- With release 10.0.0, i removed the validation module ``validators.py`` to avoid duplicating the work of what is now `the canonical feed validator <https://github.com/MobilityData/gtfs-validator>`_ (written in Java).
- I'll probably move to Polars + Polars ST for the next big release so the library can handle big feeds faster.

Maintainer Notes
================
- Before pushing to master, export the example Marimo notebook to Jupyter via ``uv run marimo export ipynb notebooks/examples.py -o notebooks/examples.ipynb --include-outputs -f``, because the docs refer to that Github-displayable version.
- After pushing to master, update the published docs via ``uv run make -C docs publish-docs-github``