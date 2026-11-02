import csv
import pytest
from pathlib import Path
from ablation_runner import (
    CONDITIONS,
    _deep_update,
    extract_metrics,
    run_ablation_study,
)


def test_ablation_conditions_definitions():
    ids = [c["id"] for c in CONDITIONS]
    assert ids == ["A0", "A1", "A2", "A3", "A4", "A5", "L0", "B0", "B1", "B2", "B3", "B4", "B5", "B6"]

    cond_map = {c["id"]: c for c in CONDITIONS}

    # A0: no pipeline run
    assert cond_map["A0"]["run_pipeline"] is False

    # A1: zero transfer and no GPU
    a1_cfg = cond_map["A1"]["overrides"]["fusion"]
    assert a1_cfg["adaptive_protection"]["max_visible_transfer"] == 0.0
    assert a1_cfg["adaptive_protection"]["max_occluded_transfer"] == 0.0
    assert a1_cfg["optimization"]["enabled"] is False
    assert a1_cfg["smoothing"]["enabled"] is True

    # A2: transfer enabled, no GPU
    a2_cfg = cond_map["A2"]["overrides"]["fusion"]
    assert a2_cfg["adaptive_protection"]["enabled"] is True
    assert a2_cfg["optimization"]["enabled"] is False

    # A3: full adaptive
    a3_cfg = cond_map["A3"]["overrides"]["fusion"]
    assert a3_cfg["adaptive_protection"]["enabled"] is True
    assert a3_cfg["optimization"]["enabled"] is True

    # A4: strong alignment downweight
    a4_cfg = cond_map["A4"]["overrides"]["fusion"]
    assert a4_cfg["adaptive_protection"]["transform_scale_m"] == 0.05
    assert a4_cfg["optimization"]["enabled"] is True

    # A5: without excess-jitter penalty
    a5_cfg = cond_map["A5"]["overrides"]["fusion"]
    assert a5_cfg["adaptive_protection"]["jitter_floor_m"] == 999.0
    assert a5_cfg["optimization"]["enabled"] is True

    # L0: legacy path
    l0_cfg = cond_map["L0"]["overrides"]["fusion"]
    assert l0_cfg["adaptive_protection"]["enabled"] is False
    assert l0_cfg["master_protection"]["enabled"] is True
    assert l0_cfg["optimization"]["enabled"] is True
    for label in ("A1", "A2", "A3", "A4", "A5"):
        old_cfg = cond_map[label]["overrides"]["fusion"]
        assert old_cfg["adaptive_protection"]["bone_quality"]["enabled"] is False
        assert old_cfg["adaptive_protection"]["risk_aware_enabled"] is False
        assert old_cfg["smoothing"]["motion_aware"]["enabled"] is False
        assert old_cfg["optimization"]["adaptive_objectives_enabled"] is False
    assert cond_map["B0"]["overrides"]["fusion"]["adaptive_protection"]["risk_aware_enabled"] is True
    assert cond_map["B1"]["overrides"]["fusion"]["adaptive_protection"]["bone_quality"]["enabled"] is False
    assert cond_map["B2"]["overrides"]["fusion"]["adaptive_protection"]["max_occluded_transfer"] == 0.0
    assert cond_map["B3"]["overrides"]["fusion"]["adaptive_protection"]["visible_risk_enabled"] is False
    assert cond_map["B4"]["overrides"]["fusion"]["smoothing"]["motion_aware"]["enabled"] is False
    assert cond_map["B5"]["overrides"]["fusion"]["optimization"]["adaptive_objectives_enabled"] is False
    assert cond_map["B6"]["overrides"]["fusion"]["bone_output"]["reference_mode"] == "robust_median"


def test_deep_update():
    base = {
        "fusion": {
            "adaptive_protection": {"enabled": True, "scale": 0.20},
            "smoothing": {"enabled": True},
        }
    }
    overrides = {
        "fusion": {
            "adaptive_protection": {"scale": 0.05},
            "optimization": {"enabled": False},
        }
    }
    merged = _deep_update(base, overrides)
    assert merged["fusion"]["adaptive_protection"]["enabled"] is True
    assert merged["fusion"]["adaptive_protection"]["scale"] == 0.05
    assert merged["fusion"]["smoothing"]["enabled"] is True
    assert merged["fusion"]["optimization"]["enabled"] is False


def test_extract_metrics_mock(tmp_path):
    mpjpe_file = tmp_path / "MPJPE_cam1.csv"
    pa_file = tmp_path / "PA-MPJPE_cam1.csv"
    accel_file = tmp_path / "Accel_cam1.csv"
    mble_file = tmp_path / "MBLE_cam1.csv"

    with open(mpjpe_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "posed_priority1_mm", "fused_priority1_mm", "posed_occ_mm", "fused_occ_mm"])
        w.writerow(["1", "60.0", "50.0", "70.0", "55.0"])
        w.writerow(["AVERAGE", "60.0", "50.0", "70.0", "55.0"])

    with open(pa_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "posed_priority1_mm", "fused_priority1_mm", "posed_vis_mm", "fused_vis_mm"])
        w.writerow(["1", "55.0", "45.0", "50.0", "42.0"])
        w.writerow(["AVERAGE", "55.0", "45.0", "50.0", "42.0"])

    with open(accel_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Evaluated_Camera", "Ground_Truth_Camera", "Module", "Frame_Accel_Error_mm_frame2"])
        w.writerow(["1", "camera1", "camera2", "posed", "65.5"])
        w.writerow(["1", "camera1", "camera2", "fused", "14.2"])

    with open(mble_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Evaluated_Camera", "Ground_Truth_Camera", "Module", "Frame_MBLE_mm"])
        w.writerow(["1", "camera1", "camera2", "posed", "13.0"])
        w.writerow(["1", "camera1", "camera2", "fused", "12.0"])

    posed = extract_metrics(tmp_path, prefix="posed")
    assert posed["mpjpe"] == 60.0
    assert posed["pa_mpjpe"] == 55.0
    assert posed["accel"] == 65.5
    assert posed["mble"] == 13.0

    fused = extract_metrics(tmp_path, prefix="fused")
    assert fused["mpjpe"] == 50.0
    assert fused["pa_mpjpe"] == 45.0
    assert fused["accel"] == 14.2
    assert fused["mble"] == 12.0


def test_ablation_runner_dry_run(tmp_path):
    csv_out = tmp_path / "results.csv"
    md_out = tmp_path / "report.md"
    results = run_ablation_study(
        out_csv=str(csv_out),
        out_md=str(md_out),
        dry_run=True,
    )
    assert len(results) == len(CONDITIONS)
    assert csv_out.exists()
    assert md_out.exists()

