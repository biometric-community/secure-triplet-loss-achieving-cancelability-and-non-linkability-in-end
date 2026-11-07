"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re


def eer(report: str, dataset: str) -> float:
    match = re.search(rf"\| {dataset} \| [\d.]+% \| ([\d.]+)% \|", report)
    assert match, f"{dataset} row missing from REPORT.md"
    return float(match.group(1))


def test_verification_eer_matches_the_face_results(report, targets):
    """Sec. V-A: 13.61% EER, about 2 points higher with the linkability term."""
    assert eer(report, "lfw") <= targets["face_eer"]["with_linkability_max"]


def test_dsys_is_reported(report):
    """Sec. IV-C: Dsys is the main metric for template linkability."""
    assert re.search(r"D.?sys", report)


def test_linkability_gap_is_small(report):
    """Sec. III-C: mated and non-mated key-mismatched scores should be similar."""
    match = re.search(r"mean gap: \*\*([\d.]+)\*\*", report)
    assert match and float(match.group(1)) <= 0.1
