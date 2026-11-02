# Ablation Study Runner for Monocular Optimization Pipeline
# Evaluates conditions A0, A1, A2, A3, A4, A5, and L0.
# Outputs to output/ablation/<condition_id>/ and root summary.
# Zero-GT compliant: Ground truth is isolated strictly to the evaluation module.

import argparse
import copy
import csv
import itertools
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import yaml

from config_loader import load_config, absolutize_config_paths
from preprocess_pipeline.executor import run_preprocess
from pose_pipeline.executor import run_pose_export
from fusion_pipeline.executor import run_fusion
from evaluation_pipeline.executor import run_evaluation

WS_DIR = Path(__file__).parent.resolve()
DEFAULT_ABLATION_DIR = WS_DIR / "output" / "ablation"

CONDITIONS = [
    {
        "id": "A0",
        "name": "Posed baseline",
        "adaptive": "—",
        "joint_transfer": "Không",
        "smoothing": "Không",
        "gpu": "Không",
        "diff_vs_control": "Dùng cột posed_* của cùng evaluation, không chạy thêm pipeline",
        "run_pipeline": False,
        "overrides": {},
    },
    {
        "id": "A1",
        "name": "Smoothing-only",
        "adaptive": "Bật",
        "joint_transfer": "Ép trọng số về 0",
        "smoothing": "Bật",
        "gpu": "Tắt",
        "diff_vs_control": "Cô lập tác động của smoothing + bone projection",
        "run_pipeline": True,
        "overrides": {
            "fusion": {
                "adaptive_protection": {
                    "enabled": True,
                    "max_visible_transfer": 0.0,
                    "max_occluded_transfer": 0.0,
                },
                "smoothing": {"enabled": True},
                "optimization": {"enabled": False},
            }
        },
    },
    {
        "id": "A2",
        "name": "Adaptive, no GPU",
        "adaptive": "Bật",
        "joint_transfer": "Bật",
        "smoothing": "Bật",
        "gpu": "Tắt",
        "diff_vs_control": "So A2 − A1 để đo riêng việc đưa thông tin camera phụ vào",
        "run_pipeline": True,
        "overrides": {
            "fusion": {
                "adaptive_protection": {
                    "enabled": True,
                    "max_visible_transfer": 0.55,
                    "max_occluded_transfer": 0.90,
                },
                "smoothing": {"enabled": True},
                "optimization": {"enabled": False},
            }
        },
    },
    {
        "id": "A3",
        "name": "Full Adaptive",
        "adaptive": "Bật",
        "joint_transfer": "Bật",
        "smoothing": "Bật",
        "gpu": "Bật",
        "diff_vs_control": "So A3 − A2 để đo ảnh hưởng GPU optimizer",
        "run_pipeline": True,
        "overrides": {
            "fusion": {
                "adaptive_protection": {
                    "enabled": True,
                    "transform_scale_m": 0.20,
                    "jitter_floor_m": 0.025,
                    "max_visible_transfer": 0.55,
                    "max_occluded_transfer": 0.90,
                },
                "smoothing": {"enabled": True},
                "optimization": {"enabled": True},
            }
        },
    },
    {
        "id": "A4",
        "name": "Strong alignment downweight",
        "adaptive": "Bật",
        "joint_transfer": "Bật",
        "smoothing": "Bật",
        "gpu": "Bật",
        "diff_vs_control": "Như A3, chỉ hạ transform_scale_m từ 0,20 xuống 0,05 m",
        "run_pipeline": True,
        "overrides": {
            "fusion": {
                "adaptive_protection": {
                    "enabled": True,
                    "transform_scale_m": 0.05,
                    "jitter_floor_m": 0.025,
                    "max_visible_transfer": 0.55,
                    "max_occluded_transfer": 0.90,
                },
                "smoothing": {"enabled": True},
                "optimization": {"enabled": True},
            }
        },
    },
    {
        "id": "A5",
        "name": "Without excess-jitter penalty",
        "adaptive": "Bật",
        "joint_transfer": "Bật",
        "smoothing": "Bật",
        "gpu": "Bật",
        "diff_vs_control": "Như A3, đặt jitter_floor_m rất lớn để vô hiệu hóa phạt jitter dư",
        "run_pipeline": True,
        "overrides": {
            "fusion": {
                "adaptive_protection": {
                    "enabled": True,
                    "transform_scale_m": 0.20,
                    "jitter_floor_m": 999.0,
                    "max_visible_transfer": 0.55,
                    "max_occluded_transfer": 0.90,
                },
                "smoothing": {"enabled": True},
                "optimization": {"enabled": True},
            }
        },
    },
    {
        "id": "L0",
        "name": "Legacy V10 path",
        "adaptive": "Tắt",
        "joint_transfer": "Theo legacy",
        "smoothing": "Bật",
        "gpu": "Bật",
        "diff_vs_control": "So với A3; vẫn chịu các cổng/khóa master của legacy",
        "run_pipeline": True,
        "overrides": {
            "fusion": {
                "adaptive_protection": {
                    "enabled": False,
                },
                "master_protection": {
                    "enabled": True,
                },
                "smoothing": {"enabled": True},
                "optimization": {"enabled": True},
            }
        },
    },
]


def _deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    return str(val)


def extract_metrics(eval_dir: Path, prefix: str = "fused") -> dict:
    mpjpe_file = eval_dir / "MPJPE_cam1.csv"
    pa_file = eval_dir / "PA-MPJPE_cam1.csv"
    accel_file = eval_dir / "Accel_cam1.csv"
    mble_file = eval_dir / "MBLE_cam1.csv"

    def _get_avg(csv_path, target_col):
        if not csv_path.exists():
            return float("nan")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or target_col not in header:
                return float("nan")
            idx = header.index(target_col)
            for row in reader:
                if row and row[0] == "AVERAGE":
                    try:
                        return float(row[idx])
                    except ValueError:
                        return float("nan")
        return float("nan")

    def _get_module_avg(csv_path, target_col, mod_val):
        if not csv_path.exists():
            return float("nan")
        vals = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or target_col not in header or "Module" not in header:
                return float("nan")
            col_idx = header.index(target_col)
            mod_idx = header.index("Module")
            for row in reader:
                if row and len(row) > mod_idx and row[mod_idx] == mod_val:
                    try:
                        vals.append(float(row[col_idx]))
                    except ValueError:
                        pass
        return float(np.mean(vals)) if vals else float("nan")

    def _round(v):
        return round(float(v), 2) if v is not None and not np.isnan(v) else float("nan")

    return {
        "mpjpe": _round(_get_avg(mpjpe_file, f"{prefix}_priority1_mm")),
        "pa_mpjpe": _round(_get_avg(pa_file, f"{prefix}_priority1_mm")),
        "mpjpe_occ": _round(_get_avg(mpjpe_file, f"{prefix}_occ_mm")),
        "mpjpe_vis": _round(_get_avg(mpjpe_file, f"{prefix}_vis_mm")),
        "pa_occ": _round(_get_avg(pa_file, f"{prefix}_occ_mm")),
        "pa_vis": _round(_get_avg(pa_file, f"{prefix}_vis_mm")),
        "accel": _round(_get_module_avg(accel_file, "Frame_Accel_Error_mm_frame2", prefix)),
        "mble": _round(_get_module_avg(mble_file, "Frame_MBLE_mm", prefix)),
    }


def _run_single_pair(base_cfg, camA, camB, gt_dir, ws_dir, cond_list, ablation_dir, dry_run=False):
    pair_id = f"{camA['id']}->{camB['id']}"
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("inputs", {})
    cfg["inputs"].update({
        "ground_truth_dir": str(gt_dir),
        "cam1_pkl": str(camA["pkl"]),
        "camera1_video": str(camA.get("video", "")),
        "cam2_pkl": str(camB["pkl"]),
        "camera2_video": str(camB.get("video", "")),
    })
    cfg["visualization"]["enabled"] = False
    cfg["runtime"]["clean_output"] = True
    cfg["learnable"]["enabled"] = False
    cfg["learnable_extra"]["enabled"] = False

    # Isolate intermediate pair outputs per pair to eliminate race conditions
    pair_tag = f"{camA['id']}_{camB['id']}"
    cfg["paths"]["preprocess_output_dir"] = str(ablation_dir / "_preprocess" / pair_tag)
    cfg["paths"]["pose_output_dir"] = str(ablation_dir / "_pose" / pair_tag)
    cfg = absolutize_config_paths(cfg, ws_dir)

    results = []
    if dry_run:
        for cond in cond_list:
            cid = cond["id"]
            cond_dir = ablation_dir / cid
            cond_dir.mkdir(parents=True, exist_ok=True)
            r = dict(cond)
            r.update({
                "pair": pair_id, "master": camA["id"], "slave": camB["id"],
                "mpjpe": 60.0, "pa_mpjpe": 50.0, "accel": 15.0, "mble": 12.0,
                "mpjpe_occ": 65.0, "mpjpe_vis": 58.0, "pa_occ": 55.0, "pa_vis": 48.0,
                "elapsed_sec": 0.1,
            })
            results.append(r)
        return results

    # Run Preprocess & Pose Export ONCE for this camera pair
    run_preprocess(cfg, extract_frames=False)
    run_pose_export(cfg)

    a0_res = None
    for cond in cond_list:
        cid = cond["id"]
        cond_dir = ablation_dir / cid
        cond_dir.mkdir(parents=True, exist_ok=True)

        c_item = dict(cond)
        c_item.update({"pair": pair_id, "master": camA["id"], "slave": camB["id"]})

        if not cond["run_pipeline"]:
            results.append(c_item)
            continue

        c_cfg = copy.deepcopy(cfg)
        _deep_update(c_cfg, cond["overrides"])
        c_cfg["paths"]["fused_output_dir"] = str(cond_dir / "fused_results")
        c_cfg["paths"]["evaluation_output_dir"] = str(cond_dir / "evaluation_results")

        t0 = time.time()
        run_fusion(c_cfg)
        run_evaluation(c_cfg)
        elapsed = time.time() - t0

        eval_dir = Path(c_cfg["paths"]["evaluation_output_dir"])
        metrics = extract_metrics(eval_dir, prefix="fused")
        c_item.update(metrics)
        c_item["elapsed_sec"] = round(elapsed, 2)
        results.append(c_item)

        if a0_res is None:
            p_metrics = extract_metrics(eval_dir, prefix="posed")
            a0_res = dict(c_item)
            a0_res.update({
                "id": "A0",
                "name": "Posed baseline",
                "adaptive": "—",
                "joint_transfer": "Không",
                "smoothing": "Không",
                "gpu": "Không",
                "diff_vs_control": "Dùng cột posed_* của cùng evaluation, không chạy thêm pipeline",
                "elapsed_sec": 0.0,
            })
            a0_res.update(p_metrics)

    for idx, r in enumerate(results):
        if r["id"] == "A0" and "pa_mpjpe" not in r:
            if a0_res:
                results[idx] = copy.deepcopy(a0_res)

    shutil.rmtree(cfg["paths"]["preprocess_output_dir"], ignore_errors=True)
    shutil.rmtree(cfg["paths"]["pose_output_dir"], ignore_errors=True)

    return results


def _write_markdown_report(doc_path: Path, results: list, base_cfg: dict, is_multi_pair: bool = False):
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    target_person = base_cfg.get("inputs", {}).get("target_person", "N/A")

    if not is_multi_pair:
        res_map = {r["id"]: r for r in results}
        a0 = res_map.get("A0", {})
        a1 = res_map.get("A1", {})
        a2 = res_map.get("A2", {})
        a3 = res_map.get("A3", {})
        a4 = res_map.get("A4", {})
        a5 = res_map.get("A5", {})
        l0 = res_map.get("L0", {})

        cam1_p = base_cfg.get("inputs", {}).get("cam1_pkl", "N/A")
        cam2_p = base_cfg.get("inputs", {}).get("cam2_pkl", "N/A")

        md = [
            "# Báo Cáo Nghiên Cứu Ablation Study (Zero-GT Adaptive Fusion Pipeline)",
            f"\n- **Thời gian chạy**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- **Đối tượng**: `{target_person}`",
            f"- **Camera 1 (Master)**: `{Path(cam1_p).name}`",
            f"- **Camera 2 (Slave)**: `{Path(cam2_p).name}`",
            "- **Nguyên tắc**: Zero-GT hoàn toàn, không sử dụng nhãn Ground Truth trong suy diễn/fusion.\n",
            "## 1. Bảng Tổng Hợp Kết Quả Thực Nghiệm",
            "\n| ID | Tên | Adaptive | Joint transfer | Smoothing | GPU | PA-MPJPE (mm) | MPJPE (mm) | Accel Error | MBLE | PA-Occ (mm) | PA-Vis (mm) | Khác với nhóm đối chứng |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|",
        ]

        for r in results:
            md.append(
                f"| **{r['id']}** | {r['name']} | {r['adaptive']} | {r['joint_transfer']} | "
                f"{r['smoothing']} | {r['gpu']} | **{_fmt(r.get('pa_mpjpe'))}** | "
                f"{_fmt(r.get('mpjpe'))} | {_fmt(r.get('accel'))} | {_fmt(r.get('mble'))} | "
                f"{_fmt(r.get('pa_occ'))} | {_fmt(r.get('pa_vis'))} | {r['diff_vs_control']} |"
            )

        md.append("\n## 2. Đo Lường Đóng Góp Từng Thành Phần (Delta Analysis)\n")

        def _diff(m, id_a, id_b):
            va = res_map.get(id_a, {}).get(m)
            vb = res_map.get(id_b, {}).get(m)
            if va is not None and vb is not None and not np.isnan(va) and not np.isnan(vb):
                return f"{va - vb:+.2f} mm"
            return "N/A"

        if a1 and a2:
            md.append("### 2.1. Đóng góp của thông tin camera phụ (A2 − A1)")
            md.append(f"- **Δ PA-MPJPE (A2 − A1)**: `{_diff('pa_mpjpe', 'A2', 'A1')}`")
            md.append(f"- **Δ PA-MPJPE Occluded**: `{_diff('pa_occ', 'A2', 'A1')}`\n")
        if a2 and a3:
            md.append("### 2.2. Ảnh hưởng của GPU Sequence Optimizer (A3 − A2)")
            md.append(f"- **Δ PA-MPJPE (A3 − A2)**: `{_diff('pa_mpjpe', 'A3', 'A2')}`")
            md.append(f"- **Δ Accel Error (A3 − A2)**: `{_diff('accel', 'A3', 'A2')}`\n")
        if a0 and a3:
            md.append("### 2.3. Tổng cải thiện mô hình Full Adaptive so với Posed Baseline (A3 − A0)")
            md.append(f"- **Δ PA-MPJPE (A3 − A0)**: `{_diff('pa_mpjpe', 'A3', 'A0')}`")
            md.append(f"- **Δ Accel Error (A3 − A0)**: `{_diff('accel', 'A3', 'A0')}`\n")
        if l0 and a3:
            md.append("### 2.4. So sánh Full Adaptive với Legacy V10 Path (A3 − L0)")
            md.append(f"- **Δ PA-MPJPE (A3 − L0)**: `{_diff('pa_mpjpe', 'A3', 'L0')}`\n")

    else:
        # Multi-pair macro statistics
        cond_ids = [c["id"] for c in CONDITIONS]
        macro = {}
        for cid in cond_ids:
            items = [r for r in results if r["id"] == cid and not np.isnan(float(r.get("pa_mpjpe", "nan")))]
            if items:
                macro[cid] = {
                    "count": len(items),
                    "mean_pa": np.mean([float(r["pa_mpjpe"]) for r in items]),
                    "mean_mpjpe": np.mean([float(r["mpjpe"]) for r in items]),
                    "mean_accel": np.mean([float(r["accel"]) for r in items]),
                    "mean_mble": np.mean([float(r["mble"]) for r in items]),
                    "mean_pa_occ": np.mean([float(r["pa_occ"]) for r in items]),
                    "mean_pa_vis": np.mean([float(r["pa_vis"]) for r in items]),
                }

        unique_pairs = sorted(list(set(r["pair"] for r in results if "pair" in r)))
        md = [
            "# Báo Cáo Nghiên Cứu Ablation Study Toàn Bộ Cặp Camera (Zero-GT Adaptive Fusion)",
            f"\n- **Thời gian chạy**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- **Đối tượng**: `{target_person}`",
            f"- **Tổng số cặp camera**: `{len(unique_pairs)}`",
            f"- **Thư mục lưu kết quả**: `output/ablation/<A0..L0>/`",
            "- **Nguyên tắc**: Zero-GT hoàn toàn, không sử dụng nhãn Ground Truth trong suy diễn/fusion.\n",
            "## 1. Bảng Thống Kê Trung Bình Toàn Bộ Cặp (Macro Averages)",
            "\n| ID | Tên | Adaptive | Joint transfer | Smoothing | GPU | Mean PA-MPJPE (mm) | Mean MPJPE (mm) | Mean Accel | Mean MBLE | Mean PA-Occ (mm) | Mean PA-Vis (mm) | Khác với nhóm đối chứng |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|",
        ]

        cond_meta = {c["id"]: c for c in CONDITIONS}
        for cid in cond_ids:
            st = macro.get(cid, {})
            cm = cond_meta.get(cid, {})
            md.append(
                f"| **{cid}** | {cm.get('name', cid)} | {cm.get('adaptive', '')} | {cm.get('joint_transfer', '')} | "
                f"{cm.get('smoothing', '')} | {cm.get('gpu', '')} | **{_fmt(st.get('mean_pa'))}** | "
                f"{_fmt(st.get('mean_mpjpe'))} | {_fmt(st.get('mean_accel'))} | {_fmt(st.get('mean_mble'))} | "
                f"{_fmt(st.get('mean_pa_occ'))} | {_fmt(st.get('mean_pa_vis'))} | {cm.get('diff_vs_control', '')} |"
            )

        md.append("\n## 2. Đo Lường Đóng Góp Vĩ Mô (Macro Delta Analysis)\n")
        def _macro_diff(cid_a, cid_b, key="mean_pa"):
            va = macro.get(cid_a, {}).get(key)
            vb = macro.get(cid_b, {}).get(key)
            if va is not None and vb is not None:
                return f"{va - vb:+.2f} mm"
            return "N/A"

        md.append(f"- **Đóng góp camera phụ (A2 − A1)**: Mean Δ PA-MPJPE = `{_macro_diff('A2', 'A1')}`")
        md.append(f"- **Ảnh hưởng GPU Sequence Optimizer (A3 − A2)**: Mean Δ PA-MPJPE = `{_macro_diff('A3', 'A2')}` | Δ Accel = `{_macro_diff('A3', 'A2', 'mean_accel')}`")
        md.append(f"- **Tổng cải thiện so với Posed Baseline (A3 − A0)**: Mean Δ PA-MPJPE = `{_macro_diff('A3', 'A0')}` | Δ Accel = `{_macro_diff('A3', 'A0', 'mean_accel')}`")
        md.append(f"- **Hiệu quả so với Legacy V10 Path (A3 − L0)**: Mean Δ PA-MPJPE = `{_macro_diff('A3', 'L0')}`")
        md.append(f"- **Hạ transform_scale_m (A4 vs A3)**: Mean Δ PA-MPJPE = `{_macro_diff('A4', 'A3')}`")
        md.append(f"- **Tắt phạt jitter dư (A5 vs A3)**: Mean Δ PA-MPJPE = `{_macro_diff('A5', 'A3')}`\n")

        md.append("## 3. Bảng Chi Tiết Từng Cặp Camera (Per-Pair Results)\n")
        md.append("| Cặp Cam | ID | PA-MPJPE (mm) | MPJPE (mm) | Accel | MBLE | PA-Occ (mm) | PA-Vis (mm) |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in results:
            p = r.get("pair", "N/A")
            md.append(f"| `{p}` | **{r['id']}** | {_fmt(r.get('pa_mpjpe'))} | {_fmt(r.get('mpjpe'))} | {_fmt(r.get('accel'))} | {_fmt(r.get('mble'))} | {_fmt(r.get('pa_occ'))} | {_fmt(r.get('pa_vis'))} |")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def run_ablation_study(
    config_path: str = "configs/pipeline.yml",
    pairs_config: str = None,
    all_pairs: bool = False,
    run_only: list = None,
    ablation_dir: str = "output/ablation",
    out_csv: str = None,
    out_md: str = None,
    max_pairs: int = None,
    dry_run: bool = False,
):
    base_cfg = load_config(WS_DIR / config_path)
    base_cfg["visualization"]["enabled"] = False
    base_cfg["runtime"]["clean_output"] = True
    base_cfg["learnable"]["enabled"] = False
    base_cfg["learnable_extra"]["enabled"] = False

    cond_list = CONDITIONS
    if run_only:
        selected = set(c.upper() for c in run_only)
        cond_list = [c for c in CONDITIONS if c["id"] in selected]

    abl_dir = Path(ablation_dir)
    if not abl_dir.is_absolute():
        abl_dir = WS_DIR / abl_dir
    abl_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(out_csv) if out_csv else (abl_dir / "results.csv")
    if not csv_path.is_absolute():
        csv_path = WS_DIR / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    md_path = Path(out_md) if out_md else (abl_dir / "report.md")
    if not md_path.is_absolute():
        md_path = WS_DIR / md_path

    # Resolve camera pairs
    pairs = []
    gt_dir = base_cfg["inputs"]["ground_truth_dir"]
    if all_pairs or pairs_config:
        p_cfg_path = WS_DIR / (pairs_config or "configs/brute_force.yml")
        with open(p_cfg_path, "r", encoding="utf-8") as f:
            bf_data = yaml.safe_load(f)
        for seg in bf_data.get("segments", []):
            cams = seg.get("cameras", [])
            s_gt = seg.get("ground_truth_dir", gt_dir)
            for cA, cB in itertools.permutations(cams, 2):
                pairs.append((cA, cB, s_gt))
        if max_pairs:
            pairs = pairs[:max_pairs]
    else:
        # Single pair from pipeline.yml
        cam1 = {"id": "cam1", "pkl": base_cfg["inputs"]["cam1_pkl"], "video": base_cfg["inputs"].get("camera1_video", "")}
        cam2 = {"id": "cam2", "pkl": base_cfg["inputs"]["cam2_pkl"], "video": base_cfg["inputs"].get("camera2_video", "")}
        pairs.append((cam1, cam2, gt_dir))

    print("\n" + "=" * 80)
    print(f"BẮT ĐẦU CHẠY BỘ THÍ NGHIỆM ABLATION STUDY ({len(pairs)} cặp)")
    print(f"Thư mục gốc: {abl_dir.resolve()} | Thư mục con: {[c['id'] for c in cond_list]}")
    print("=" * 80)

    fieldnames = [
        "pair", "master", "slave", "id", "name", "adaptive", "joint_transfer", "smoothing", "gpu",
        "pa_mpjpe", "mpjpe", "accel", "mble", "pa_occ", "pa_vis", "mpjpe_occ", "mpjpe_vis",
        "elapsed_sec", "diff_vs_control"
    ]

    all_results = []
    # Resume check if multi-pair: only consider a pair completed if it has all conditions
    completed_pairs = set()
    if all_pairs and csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                pair_counts = {}
                rows = []
                for r in reader:
                    p = r.get("pair")
                    if p:
                        rows.append(r)
                        pair_counts[p] = pair_counts.get(p, 0) + 1
                completed_pairs = {p for p, cnt in pair_counts.items() if cnt >= len(cond_list)}
                all_results = [r for r in rows if r.get("pair") in completed_pairs]
            if completed_pairs:
                print(f"[Resume] Đã tải {len(completed_pairs)} cặp đã hoàn thành từ {csv_path.name}")
        except Exception:
            pass

    for p_idx, (camA, camB, p_gt) in enumerate(pairs, 1):
        pair_key = f"{camA['id']}->{camB['id']}"
        if pair_key in completed_pairs:
            print(f">>> [{p_idx}/{len(pairs)}] Bỏ qua {pair_key} (Đã có kết quả)")
            continue

        print(f"\n>>> [{p_idx}/{len(pairs)}] Đang chạy cặp: {pair_key} ...")
        t_pair0 = time.time()
        pair_res = _run_single_pair(base_cfg, camA, camB, p_gt, WS_DIR, cond_list, abl_dir, dry_run=dry_run)
        elapsed_pair = time.time() - t_pair0

        all_results.extend(pair_res)

        # Print pair summary
        a0 = next((r for r in pair_res if r["id"] == "A0"), {})
        a3 = next((r for r in pair_res if r["id"] == "A3"), {})
        pa0 = _fmt(a0.get('pa_mpjpe'))
        pa3 = _fmt(a3.get('pa_mpjpe'))
        d_pa = f"{float(a3.get('pa_mpjpe', 0)) - float(a0.get('pa_mpjpe', 0)):+.2f} mm" if 'pa_mpjpe' in a0 and 'pa_mpjpe' in a3 else "N/A"
        print(f"    --> [{pair_key}] A0={pa0} mm | A3={pa3} mm (Δ={d_pa}) [{elapsed_pair:.1f}s]")

        # Stream incrementally to root CSV
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        # Stream incrementally to per-condition CSVs (output/ablation/<cid>/results.csv)
        for cond in cond_list:
            cid = cond["id"]
            cond_csv = abl_dir / cid / "results.csv"
            cond_rows = [r for r in all_results if r["id"] == cid]
            if cond_rows:
                with open(cond_csv, "w", newline="", encoding="utf-8-sig") as f:
                    c_writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    c_writer.writeheader()
                    c_writer.writerows(cond_rows)

    # Save Markdown reports (root and docs)
    _write_markdown_report(md_path, all_results, base_cfg, is_multi_pair=(len(pairs) > 1 or all_pairs))
    docs_report = WS_DIR / "docs" / "ablation_study_report.md"
    _write_markdown_report(docs_report, all_results, base_cfg, is_multi_pair=(len(pairs) > 1 or all_pairs))

    print("\n" + "=" * 80)
    print("HOÀN TẤT THỰC NGHIỆM ABLATION STUDY")
    print(f"  - Thư mục ablation: {abl_dir.resolve()}")
    print(f"  - CSV report:       {csv_path.resolve()}")
    print(f"  - Markdown report:  {md_path.resolve()}")
    print("=" * 80)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monocular Optimization Ablation Study")
    parser.add_argument("--config", type=str, default="configs/pipeline.yml", help="Base config path")
    parser.add_argument("--ablation-dir", type=str, default="output/ablation", help="Base output directory for ablations")
    parser.add_argument("--all-pairs", action="store_true", help="Run ablation across all pairs in brute_force.yml")
    parser.add_argument("--pairs-config", type=str, default=None, help="Custom pairs config path")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit number of pairs")
    parser.add_argument("--run-only", type=str, default=None, help="Comma-separated IDs to run, e.g. A0,A1,A3")
    parser.add_argument("--out-csv", type=str, default=None, help="Output CSV path (default: <ablation-dir>/results.csv)")
    parser.add_argument("--out-md", type=str, default=None, help="Output MD path (default: <ablation-dir>/report.md)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without executing pipeline")
    args = parser.parse_args()

    run_list = [s.strip() for s in args.run_only.split(",")] if args.run_only else None
    run_ablation_study(
        config_path=args.config,
        pairs_config=args.pairs_config,
        all_pairs=args.all_pairs,
        run_only=run_list,
        ablation_dir=args.ablation_dir,
        out_csv=args.out_csv,
        out_md=args.out_md,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
    )
