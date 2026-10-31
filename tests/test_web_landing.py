"""Tests for the ORBIT-PRISM web page in web/ (a single working application page, offline)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def _html():
    return (WEB / "index.html").read_text(encoding="utf-8")


def test_page_has_the_sections_the_demo_needs():
    html = _html()
    for sec in ("top", "problem", "how", "try", "sensors", "link", "results", "end"):
        assert f'id="{sec}"' in html, f"Section id='{sec}' is missing from web/index.html"


def test_problem_and_solution_are_explained():
    html = _html()
    assert "The problem" in html and "Our solution" in html
    for feature in ("Miss-rate dial", "Radar fallback", "Byte-budget scheduler"):
        assert feature in html, f"{feature} is not explained on the page"


def test_judge_qa_and_glossary_are_not_on_the_page():
    """The page is a demo of the product: the judge Q&A and glossary live in the speech, not on the site."""
    html = _html()
    assert 'id="questions"' not in html and 'id="glossary"' not in html


def test_no_invented_telemetry_or_unmeasured_claims():
    """Nothing on the page may be a made-up live reading or a claim with no measurement behind it."""
    text = _html() + (WEB / "js" / "app.js").read_text(encoding="utf-8")
    text += (WEB / "js" / "three-scene.js").read_text(encoding="utf-8")
    for bad in ("Gbps", "SATELLITE ONLINE", "NADIR LOCK", "4.8×", "under 90 seconds", "hudLat", "hudLon"):
        assert bad not in text, f"web/ contains {bad!r}"
    assert "Math.random" not in (WEB / "js" / "app.js").read_text(encoding="utf-8")


def test_downlink_animation_elements_are_built_by_the_simulator_view():
    js = (WEB / "js" / "app.js").read_text(encoding="utf-8")
    for el in ("playQueueBtn", "budgetProgress", "budgetVal", "passBars", "budgetPct"):
        assert f'id="{el}"' in js, f"{el} is not rendered by the link simulator"


def test_no_external_network_requests():
    """The site works offline: no CDN, no web fonts, no http/https sources."""
    for file in [WEB / "index.html", *WEB.glob("css/*.css"), *WEB.glob("js/*.js")]:
        text = file.read_text(encoding="utf-8")
        assert not re.search(r'(?:src|href)=["\']https?://', text), f"{file.name} loads from internet"
        assert "@import" not in text, f"{file.name} uses @import"


def test_no_banned_claims_in_web():
    terms = ["out" + "perform", "guar" + "antee", "real-time on " + "satellite", "danger " + "score",
             "disaster " + "management system"]
    banned = re.compile("|".join(terms), re.I)
    for file in [WEB / "index.html", *WEB.glob("js/*.js")]:
        for n, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            assert not banned.search(line), f"{file.name}:{n} contains banned claim: {line.strip()}"


def test_scaled_pass_explicitly_labelled():
    assert "scaled pass" in _html().lower()


def test_layer_names_match_the_packet_definition():
    """L0 is the 32-byte alert, L1 the 2 KiB preview, L2 the full tile (src/constants.py)."""
    text = _html() + (WEB / "js" / "app.js").read_text(encoding="utf-8")
    assert "L0 represents full" not in text and "L2 represents immediate" not in text


def test_3d_background_is_loaded_from_local_files_and_answers_the_cursor():
    html = _html()
    for script in ("js/vendor/three.min.js", "js/three-scene.js"):
        assert script in html and (WEB / script).exists()
    scene = (WEB / "js" / "three-scene.js").read_text(encoding="utf-8")
    assert "pointermove" in scene and "journeyProgress" in scene and "setChip3D" in scene
    for tex in ("earth_atmos_2048.jpg", "earth_lights_2048.png", "earth_clouds_1024.png"):
        assert tex in scene and (WEB / "assets" / "earth" / tex).exists()
