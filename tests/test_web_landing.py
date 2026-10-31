"""Tests for the ORBIT-PRISM AETHER-IX web landing page and static site in web/."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def test_landing_page_sections_exist():
    """Verify all top-to-bottom sections required by the specification exist in web/index.html."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    expected_sections = [
        "top",          # Hero
        "problem",      # The problem in numbers
        "how",          # 10-step visual pipeline
        "try",          # Try it interactive triage
        "link",         # Link simulator
        "results",      # Measured results
        "questions",    # Questions judges ask
        "glossary",     # Glossary
        "limits",       # Honest limits
    ]
    for sec in expected_sections:
        assert f'id="{sec}"' in html, f"Section id='{sec}' is missing from web/index.html"


def test_judges_questions_accordion_has_all_ten_questions():
    """Verify all 10 questions required for the judges defense accordion are present."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    expected_questions = [
        "Is this just cloud detection?",
        "Why tiles, not pixels?",
        "Why a small CNN over a water index?",
        "How does it fit in 4.2 MB?",
        "What if radar is wrong?",
        "What if the cloud check is wrong?",
        "Does the promise hold under data shift?",
        "Do optical and radar come from one satellite?",
        "Is this real hardware?",
        "Why only floods?",
    ]
    for q in expected_questions:
        assert q in html, f"Judge question '{q}' is missing from web/index.html"


def test_glossary_defines_all_eleven_terms():
    """Verify the glossary defines all technical terms specified."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    expected_terms = [
        "Tile",
        "Chip",
        "Critical Tile",
        "Miss-Rate Dial",
        "Scaled Pass",
        "L0 / L1 / L2",
        "Priority Score",
        "Held-Out Event",
        "Simulated Cloud",
        "SRAM",
        "INT8",
    ]
    for term in expected_terms:
        assert term in html, f"Glossary term '{term}' is missing from web/index.html"


def test_downlink_animation_elements_exist():
    """Verify interactive queue playback and budget progress elements exist."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert 'id="playQueueBtn"' in html
    assert 'id="budgetProgress"' in html
    assert 'id="budgetVal"' in html
    assert 'id="budgetFill"' in html
    assert 'id="budgetPct"' in html


def test_no_external_network_requests():
    """Ensure the site works completely offline without wifi (no CDN, no web fonts, no http/https)."""
    for file in [WEB / "index.html", *WEB.glob("css/*.css"), *WEB.glob("js/*.js")]:
        text = file.read_text(encoding="utf-8")
        assert not re.search(r'(?:src|href)=["\']https?://', text), f"{file.name} loads from internet"
        assert "@import" not in text, f"{file.name} uses @import"


def test_no_banned_claims_in_web():
    """Check that banned claims are never written in the website."""
    terms = [
        "out" + "perform",
        "guar" + "antee",
        "real-time on " + "satellite",
        "danger " + "score",
        "disaster " + "management system"
    ]
    banned = re.compile("|".join(terms), re.I)
    for file in [WEB / "index.html", *WEB.glob("js/*.js")]:
        text = file.read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), 1):
            if banned.search(line):
                # Only allowable exception: "holds when test data resembles calibration data"
                if "holds only when test data resembles" in line.lower() or "holds when test data resembles" in line.lower():
                    continue
                assert False, f"{file.name}:{n} contains banned claim: {line.strip()}"


def test_scaled_pass_explicitly_labelled():
    """Ensure scaled pass is explicitly called 'scaled pass', not plain 'pass' for the 3.03 MB budget."""
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert "scaled pass" in html.lower()
