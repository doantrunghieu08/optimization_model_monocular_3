# Robust Multi-View 3D Human Pose Fusion via Ray-Casting Occlusion Awareness and Zero-GT Kinematic Quality Gating

Official implementation of the paper:
**"Robust Multi-View 3D Human Pose Fusion via 3D Ray-Casting Occlusion Detection and Zero-GT Kinematic Anti-Negative Transfer"**

---

## 📌 Abstract

Monocular 3D Human Pose Estimation (HPE) in unsynchronized, uncalibrated, or in-the-wild multi-camera environments frequently suffers from scale drift, depth ambiguity, and severe self-occlusions. Existing multi-view fusion approaches rely heavily on 2D reprojection loss or unconstrained global rigid transforms, often leading to depth collapse and negative transfer when fusing weak auxiliary viewpoints.

This repository introduces a robust, learning-isolated 3D pose fusion framework:

1. **3D Ray-Casting Occlusion Detection**: Casts physical optical rays from the camera center to torso mesh vertices to detect true 3D volumetric self-occlusion without 2D projective distortion.
2. **Direct 3D Kinematic Pose Fusion**: Discards fragile 2D reprojection constraints in favor of 3D skeletal space optimization with hierarchical kinematic bone rotations.
3. **Zero-GT Biomechanical Quality Gating**: Dynamically identifies master vs. slave reliability without ground-truth labels using high-frequency temporal acceleration (jitter) and tracking continuity metrics.
4. **Rigid Bone Length Locking & Temporal Smoothing**: Enforces anatomical bone length constraints via forward kinematic projection and applies Savitzky-Golay polynomial filtering along trajectory timelines.
5. **Strict Zero-Ground-Truth Protocol**: Guarantees complete isolation of test annotations during all preprocessing, alignment, and fusion stages. Ground truth is solely ingested in the evaluation pipeline.

---

## 🚀 Key Methodological Contributions

### 1. Volumetric 3D Occlusion Awareness
Joint visibility is evaluated via physical ray-surface intersection:
$$\mathbf{r}(t) = t \cdot \frac{P_k}{\|P_k\|_2}, \quad t_{\text{hit}} = \min \{ t > 0 \mid \mathbf{r}(t) \cap \mathcal{M}_{\text{torso}} \ne \emptyset \}$$
$$\text{Occluded}(P_k) \iff t_{\text{hit}} < \|P_k\|_2 - \tau \quad (\tau = 0.05\,\text{m})$$

### 2. Hierarchical Kinematic Bone Rotation
Prevents the lever-arm effect by rotating relative bone orientation vectors along kinematic sub-trees rather than forcing global rigid transforms:
$$\hat{P}_{\text{slave}\to\text{master}}(c) = P_1(p) + \mathbf{R}_{21} \cdot (P_2(c) - P_2(p))$$

### 3. Zero-GT Temporal Jitter & Continuity Detection
Overcomes the "Ray-Casting Visibility Trap" (where unoccluded distant cameras exhibit deceptively high visibility despite large 3D depth error) through intrinsic kinematic signals:
* **Temporal Jitter**: Measures non-biological high-frequency tracking acceleration:
  $$a = \frac{1}{T-2} \sum_{t=1}^{T-2} \| P_{t+1} - 2P_t + P_{t-1} \|_2$$
* **Adaptive Master Protection**: Automatically locks master margins when slave jitter is significantly higher ($a_{\text{slave}} > 1.4 a_{\text{master}}$), while dropping margins to zero when master is degraded, enabling maximal uninhibited correction.

### 4. Anatomical Bone Length Locking & Savitzky-Golay Smoothing
* Forward kinematic ray projection guarantees zero bone distortion across all fused frames ($\mathbf{100\%}$ positive MBLE improvement).
* Savitzky-Golay quadratic filtering reduces acceleration error by **over 50%**.

---

## 📊 Benchmark Results

Evaluated on the **Harmony4D / Multi-view In-the-Wild Benchmark (55 Pair Combinations)**:

| Method / View Pair | Base MPJPE (mm) | Fused MPJPE (mm) | Base PA-MPJPE (mm) | Fused PA-MPJPE (mm) | $\Delta$ PA-MPJPE (%) | Accel Error | MBLE Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monocular Baseline** | 580.76 | — | 107.37 | — | — | 27.94 | 45.12 |
| **Ours (`cam13 -> cam19`)** | 554.63 | **560.87** | 45.89 | **35.89** | **+21.79%** | **12.45** | **14.20** |
| **Ours (`cam03 -> cam19`)** | 624.51 | **620.90** | 90.27 | **73.16** | **+18.95%** | **12.83** | **14.42** |
| **Ours (`cam04 -> cam19`)** | 560.12 | **558.40** | 41.14 | **34.08** | **+17.16%** | **11.20** | **13.80** |
| **Ours (`cam19 -> cam20`)** *(Rescued)* | 520.10 | **519.80** | 27.85 | **27.50** | **+1.26%** | **10.50** | **12.90** |
| **Ours (`cam04 -> cam17`)** *(Rescued)* | 545.20 | **543.10** | 40.44 | **39.75** | **+1.71%** | **11.80** | **13.50** |
| **Ours (`cam05 -> cam17`)** *(Rescued)* | 540.30 | **538.90** | 38.87 | **38.73** | **+0.36%** | **11.40** | **13.10** |

* Overall benchmark statistics: **> 85% positive pairs**, average Accel delta **+55.36%**, and **100% positive MBLE consistency**.
* *All experiments strictly adhere to the Zero-GT isolation protocol.*

---

## 🛠️ Installation & Environment Setup

- **OS**: Windows / Linux
- **Python**: $\ge 3.10$
- **CUDA**: 12.1+ (NVIDIA GPU recommended)

```bash
git clone https://github.com/doantrunghieu08/optimization_model_monocular_3.git
cd optimization_model_monocular_3

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚡ Quickstart & Reproducibility

### 1. Dual-View Pipeline Inference
Run the primary end-to-end pipeline on configured camera pairs:
```bash
python main.py
```
*Configure inputs, SMPL model paths, and fusion parameters in [`configs/pipeline.yml`](configs/pipeline.yml).*

### 2. Multi-View Combinatorial Benchmark (Brute-Force Sweep)
Evaluate all pairwise camera permutations across sequence segments:
```bash
python brute_force_runner.py
```
Outputs comprehensive performance metrics to `output/brute_force_report.csv` and `output/brute_force_report.html`, ranked by $\% \Delta \text{PA-MPJPE}$.

### 3. Interpretable Factor Analysis
Extract dominant decision rules and feature importances governing fusion gains:
```bash
python analyze_tree.py
```

### 4. Regression & Verification Suite
Run continuous verification to ensure mathematical consistency and zero-GT isolation:
```bash
pytest tests/
```

---

## 📂 Repository Structure

```text
optimization_model_monocular_3/
├── configs/                   # Pipeline configuration and keypoint schemas
│   ├── pipeline.yml           # Core pipeline parameters
│   └── brute_force.yml        # Multi-camera evaluation sweep config
├── fusion_pipeline/           # Core library
│   ├── detector.py            # 3D ray-casting occlusion detection & bone analysis
│   ├── correction.py          # Kinematic bone rotation & RANSAC alignment
│   ├── optimization.py        # 3D constrained GPU optimizer with min-confidence loss
│   └── executor.py            # Multi-stage execution engine with Zero-GT quality gating
├── pose_pipeline/             # WHAM / SMPL pose export & frame slicing
├── preprocess_pipeline/       # 2D tracking parser, calibration & offset estimation
├── evaluation_pipeline/       # Zero-GT isolated evaluation (MPJPE, PA-MPJPE, MBLE, Accel)
├── models/                    # SMPL neutral body model & joint regressors
├── tests/                     # 36 regression & compliance unit tests
├── main.py                    # Dual-view pipeline entry point
├── brute_force_runner.py      # Exhaustive multi-view benchmark runner
├── analyze_tree.py            # Decision tree interpretable analysis script
├── MODEL_CHANGES.md           # Mathematical formulations and architectural changelog
└── compat.py                  # Backward-compatible data loading layer
```

---

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{hieu2026robustposefusion,
  title={Robust Multi-View 3D Human Pose Fusion via Ray-Casting Occlusion Awareness and Zero-GT Kinematic Quality Gating},
  author={Tien Thanh - Nguyen, Trung Hieu - Doan and Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
