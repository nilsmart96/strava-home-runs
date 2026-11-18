from datetime import datetime, timezone

import numpy as np
import pytest

from homeruns import streetmap, tracks
from homeruns.__main__ import parse_args

T0 = datetime(2024, 5, 1, 7, 0, tzinfo=timezone.utc)


def run(lat0, lon0, lat1, lon1, n=200):
    loc = np.column_stack([np.linspace(lat0, lat1, n), np.linspace(lon0, lon1, n)])
    return tracks.Track("run", T0, "running", loc, np.arange(n, dtype=float))


def test_frame_is_long_px_wide_and_reaches_the_zoom_with_street_names():
    fr = streetmap.frame([run(38.70, -9.16, 38.72, -9.13)])         # ~3 km, Lisbon
    assert max(fr["css_w"], fr["css_h"]) * streetmap.DPR == pytest.approx(8000, abs=2)
    assert fr["zoom"] >= streetmap.NAMES_ZOOM


def test_frame_grows_a_wide_map_to_keep_street_names():
    fr = streetmap.frame([run(59.90, 10.60, 59.92, 10.83)])         # ~13 km wide, Oslo
    assert streetmap.NAMES_ZOOM <= fr["zoom"] < streetmap.NAMES_ZOOM + 1e-3
    assert 8000 < fr["css_w"] * streetmap.DPR <= streetmap.MAX_PX


def test_heat_image_is_aligned_with_the_map_and_darker_where_runs_overlap():
    a, b = run(38.700, -9.160, 38.700, -9.140), run(38.704, -9.160, 38.704, -9.140)
    fr = streetmap.frame([a, b])
    im = np.asarray(streetmap.heat_image([a, a, a, b], fr), float)
    assert im.shape[:2] == (fr["css_h"] * streetmap.DPR, fr["css_w"] * streetmap.DPR)
    scale = 512 * 2 ** fr["zoom"] * streetmap.DPR
    def pixel(lat, lon):
        x, y = streetmap.merc(lat, lon)
        return im[int((y - fr["y0"]) * scale), int((x - fr["x0"]) * scale)]
    busy, quiet, empty = pixel(38.700, -9.150), pixel(38.704, -9.150), pixel(38.702, -9.150)
    assert empty[3] == 0
    assert busy[3] > quiet[3] > 0
    assert busy[:3].sum() < quiet[:3].sum()                        # darker ink on the busy street


def test_page_uses_the_style_and_puts_the_heat_under_the_labels():
    fr = streetmap.frame([run(38.70, -9.16, 38.72, -9.13)])
    html = streetmap.page_html(fr, "positron")
    assert "tiles.openfreemap.org/styles/positron" in html
    assert "layers.find(l => l.type === 'symbol')" in html


def test_find_chrome_reports_a_missing_browser():
    with pytest.raises(SystemExit):
        streetmap.find_chrome("/nonexistent/chrome")


def test_street_map_option_defaults_to_liberty_and_works_without_a_video():
    assert parse_args(["runs", "--street-map"]).street_map == "liberty"
    assert parse_args(["runs", "--street-map", "positron", "--no-video"]).street_map == "positron"
    assert parse_args(["runs"]).street_map is None


@pytest.mark.parametrize("argv, message", [
    (["runs", "--street-map", "sepia"], "invalid choice"),
    (["--overlay-only", "--street-map"], "not --overlay-only"),
    (["runs", "--no-video"], "add --stills or --street-map"),
])
def test_bad_street_map_options(argv, message, capsys):
    with pytest.raises(SystemExit):
        parse_args(argv)
    assert message in capsys.readouterr().err
