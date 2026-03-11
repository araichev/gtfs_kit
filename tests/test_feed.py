import pytest
from pathlib import Path
import shutil
import tempfile
import io
import zipfile

import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from .context import gtfs_kit, DATA_DIR
from gtfs_kit import feed as gkf
from gtfs_kit import constants as cs


def test_feed(monkeypatch):
    # -------------------------------------------------
    # Section 1: constructor basics
    # -------------------------------------------------
    feed = gkf.Feed(agency=pd.DataFrame(), dist_units="km")
    for key in cs.FEED_ATTRS:
        val = getattr(feed, key)
        if key == "dist_units":
            assert val == "km"
        elif key == "agency":
            assert isinstance(val, pd.DataFrame)
        else:
            assert val is None

    # -------------------------------------------------
    # Section 2: expected wrapped methods exist
    # -------------------------------------------------
    for name in [
        "get_dates",
        "clean",
        "describe",
        "get_routes",
        "get_shapes",
        "get_stop_times",
        "get_stops",
        "get_trips",
    ]:
        assert hasattr(feed, name)
        assert callable(getattr(feed, name))

    # -------------------------------------------------
    # Section 3: every declared feed method is attached
    # -------------------------------------------------
    for name in gkf._FEED_METHODS:
        assert hasattr(gkf.Feed, name)
        assert callable(getattr(gkf.Feed, name))

    # -------------------------------------------------
    # Section 4: wrapper delegates self/args/kwargs
    # -------------------------------------------------
    seen = {}

    def fake_get_trips(feed_arg, *args, **kwargs):
        seen["feed"] = feed_arg
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    monkeypatch.setitem(gkf._FEED_METHODS, "get_trips", fake_get_trips)
    monkeypatch.setattr(gkf.Feed, "get_trips", gkf._make_feed_method(fake_get_trips))

    result = feed.get_trips("20240101", time="08:00:00")

    assert result == "ok"
    assert seen["feed"] is feed
    assert seen["args"] == ("20240101",)
    assert seen["kwargs"] == {"time": "08:00:00"}

    # -------------------------------------------------
    # Section 5: wraps preserves basic metadata
    # -------------------------------------------------
    assert gkf.Feed.get_trips.__name__ == fake_get_trips.__name__
    assert gkf.Feed.get_trips.__doc__ == fake_get_trips.__doc__

    # -------------------------------------------------
    # Section 6: one real wrapped method still works
    # -------------------------------------------------
    sample_feed = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    result = sample_feed.subset_dates(["20070101"])
    assert isinstance(result, list)


def test_str():
    feed = gkf.Feed(agency=pd.DataFrame(), dist_units="km")
    assert isinstance(str(feed), str)


def test_eq():
    assert gkf.Feed(dist_units="m") == gkf.Feed(dist_units="m")

    feed1 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"]),
    )
    assert feed1 == feed1

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[3, 4], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 != feed2

    feed2 = gkf.Feed(
        dist_units="m",
        stops=pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"]),
    )
    assert feed1 == feed2

    feed2 = gkf.Feed(dist_units="mi", stops=feed1.stops)
    assert feed1 != feed2


def test_copy():
    feed1 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    feed2 = feed1.copy()

    # Check attributes
    for key in cs.FEED_ATTRS:
        val = getattr(feed2, key)
        expect_val = getattr(feed1, key)
        if isinstance(val, pd.DataFrame):
            assert_frame_equal(val, expect_val)
        elif isinstance(val, pd.core.groupby.DataFrameGroupBy):
            assert val.groups == expect_val.groups
        else:
            assert val == expect_val


# --------------------------------------------
# Test functions about inputs and outputs
# --------------------------------------------
def test_list_feed():
    # Bad path
    with pytest.raises(ValueError):
        gkf.list_feed("bad_path!")

    for path in [DATA_DIR / "sample_gtfs.zip", DATA_DIR / "sample_gtfs"]:
        f = gkf.list_feed(path)
        assert isinstance(f, pd.DataFrame)
        assert set(f.columns) == {"file_name", "file_size"}
        assert f.shape[0] in [12, 13]


def test_read_feed_from_url(monkeypatch):
    # Build an in-memory zip payload.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("agency.txt", "agency_id,agency_name,agency_url,agency_timezone\n")
    zip_bytes = buf.getvalue()

    class FakeResponse:
        def __init__(self, content: bytes):
            self._content = content
            self.headers = {"Content-Type": "application/octet-stream"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1024 * 1024):
            yield self._content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    captured = {}

    def fake_get(url, stream=True, timeout=(10, 60)):
        assert url == "https://example.com/feed.zip"
        assert stream is True
        assert timeout == (10, 60)
        return FakeResponse(zip_bytes)

    def fake_read_feed_from_path(path, dist_units):
        # The temp file should exist while being parsed.
        assert path.exists()
        assert zipfile.is_zipfile(path)
        captured["path"] = path
        captured["dist_units"] = dist_units
        return "FAKE_FEED"

    monkeypatch.setattr(gkf.requests, "get", fake_get)
    monkeypatch.setattr(gkf, "_read_feed_from_path", fake_read_feed_from_path)

    result = gkf.read_feed("https://example.com/feed.zip", dist_units="km")

    assert result == "FAKE_FEED"
    assert captured["dist_units"] == "km"
    # TemporaryDirectory cleanup should remove the downloaded file after return.
    assert not captured["path"].exists()


def test_read_feed():
    # Bad path
    with pytest.raises(ValueError):
        gkf.read_feed("bad_path!", dist_units="km")

    # Bad dist_units
    with pytest.raises(ValueError):
        gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="bingo")

    # Requires dist_units
    with pytest.raises(TypeError):
        gkf.read_feed(path=DATA_DIR / "sample_gtfs.zip")

    # Success
    feed = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="m")

    # Success
    feed = gkf.read_feed(DATA_DIR / "sample_gtfs", dist_units="m")

    # Feed should have None feed_info table
    assert feed.feed_info is None


def test_to_file():
    feed1 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")

    # Export feed1, import it as feed2, and then test equality
    for out_path in [DATA_DIR / "bingo.zip", DATA_DIR / "bingo"]:
        feed1.to_file(out_path)
        feed2 = gkf.read_feed(out_path, "km")
        assert feed1 == feed2
        try:
            out_path.unlink()
        except Exception:
            shutil.rmtree(str(out_path))

    # Test that integer columns with NaNs get output properly.
    feed3 = gkf.read_feed(DATA_DIR / "sample_gtfs.zip", dist_units="km")
    f = feed3.trips.copy()
    f.loc[0, "direction_id"] = np.nan
    f.loc[1, "direction_id"] = 1
    f.loc[2, "direction_id"] = 0
    feed3.trips = f
    q = DATA_DIR / "bingo.zip"
    feed3.to_file(q)

    tmp_dir = tempfile.TemporaryDirectory()
    shutil.unpack_archive(str(q), tmp_dir.name, "zip")
    qq = Path(tmp_dir.name) / "trips.txt"
    t = pd.read_csv(qq, dtype={"direction_id": "Int8"})
    assert t[~t["direction_id"].isin([pd.NA, 0, 1])].empty
    tmp_dir.cleanup()
    q.unlink()
