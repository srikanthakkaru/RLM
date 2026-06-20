"""Tests for vlmextraction freetext/heuristic parsing."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

from vlmextraction import (  # noqa: E402
    _empty_extraction,
    _extraction_has_signal,
    _merge_nondefault_extraction,
    _parse_freetext_response,
)

REPORT_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "reports" / (
    "TCGA-2E-A9G8.921E6140-A03E-4FBD-9FB8-554AE96FD16C.txt"
)


@pytest.mark.skipif(not REPORT_FIXTURE.is_file(), reason="TCGA fixture report not present")
def test_parse_freetext_tcga_report() -> None:
    text = REPORT_FIXTURE.read_text(encoding="utf-8")
    data = _parse_freetext_response(text)
    assert _extraction_has_signal(data)
    assert data["figo_grade"] == "3"
    assert data["tumor_size_cm"] == "5.2"
    assert data["myometrial_invasion_depth_cm"] == "0.5"
    assert data["lymph_nodes_total_examined"] == 28
    assert data["lymph_nodes_total_positive"] == 0
    assert data["tnm_pT"] == "pT1a"
    assert data["tnm_pN"] == "pN0"
    assert data["margin_status"] == "uninvolved"
    assert "endometrioid" in data["histologic_type"].lower()


def test_merge_nondefault_prefers_overlay() -> None:
    base = _empty_extraction()
    base["figo_grade"] = "2"
    overlay = _empty_extraction()
    overlay["figo_grade"] = "not reported"
    merged = _merge_nondefault_extraction(base, overlay)
    assert merged["figo_grade"] == "2"
