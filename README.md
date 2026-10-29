# Robust Multi-View 3D Human Pose Fusion via Ray-Casting Occlusion Awareness and Kinematic Anti-Negative Transfer

Official implementation of the paper:
**"Robust Multi-View 3D Human Pose Fusion via 3D Ray-Casting Occlusion Detection and Kinematic Anti-Negative Transfer"**

---

## 📌 Abstract

Monocular 3D Human Pose Estimation (HPE) in unsynchronized, uncalibrated, or in-the-wild multi-camera environments frequently suffers from scale drift, depth ambiguity, and severe self-occlusions. Existing multi-view fusion approaches rely heavily on 2D reprojection loss or unconstrained global rigid transforms, often leading to depth collapse and negative transfer when fusing weak auxiliary viewpoints.

This repository introduces a robust, learning-isolated 3D pose fusion framework:
1. **3D Ray-Casting Occlusion Detection**: Casts physical optical rays from the camera center to torso mesh vertices to detect true 3D volumetric self-occlusion without 2D projective distortion.
2. **Direct 3D Kinematic Pose Fusion**: Discards 2D reprojection constraints in favor of 3D skeletal space optimization with hierarchical kinematic bone rotations.
3. **Anti-Negative Transfer Gating**: Prevents performance degradation through bidirectional high-confidence locking, uncertainty gating, and min-confidence loss.
4. **Strict Zero-Ground-Truth Isolation**: Guarantees complete isolation of test annotations during all preprocessing, alignment, and fusion stages.

---

## 🚀 Key Methodological Contributions

- **Volumetric 3D Occlusion Awareness**: Evaluates joint visibility via ray-surface intersection:
  $$\mathbf{r}(t) = t \cdot \frac{P_k}{\|P_k\|_2}, \quad t_{\text{hit}} = \min \{ t > 0 \mid \mathbf{r}(t) \cap \mathcal{M}_{\text{torso}} \ne \emptyset \}$$
- **Harmonic Skeletal Belief Fusion**: Combines spatial distance penalties with structural bone-tree graph neighbor confidence.
- **Decision-Theoretic Pair Selection**: Identifies dominant camera pairs and baseline error boundaries where fusion provides positive delta gains.

---

## 📊 Benchmark Results

Evaluated on the **Harmony4D / Multi-view In-the-Wild Benchmark**:

| Method / View Pair | Base MPJPE (mm) | Fused MPJPE (mm) | Base PA-MPJPE (mm) | Fused PA-MPJPE (mm) | $\Delta$ PA-MPJPE (%) | Accel Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Monocular Baseline** | 580.76 | — | 107.37 | — | — | 27.94 |
| **Ours (cam21 + cam04)** | 580.76 | **600.96** | 107.37 | **80.32** | **+25.19%** | **16.04** |
| **Ours (cam21 + cam05)** | 581.23 | **602.79** | 106.80 | **81.80** | **+23.41%** | **16.22** |
| **Ours (cam22 + cam04)** | 554.63 | **560.87** | 64.70 | **50.36** | **+22.16%** | **9.02** |

*All experiments strictly adhere to Zero-GT isolation protocol.*

---

## 🛠️ Installation & Requirements

- **OS**: Linux / Windows
- **Python**: $\ge 3.10$
- **CUDA**: 12.1+ (NVIDIA GPU recommended)

```bash
git clone https://github.com/doantrunghieu08/optimization_model_monocular_3.git
cd optimization_model_monocular_3

pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## ⚡ Quickstart & Reproducibility

### 1. Dual-View Pipeline Inference
Run the primary end-to-end pipeline on configured camera pairs:
```bash
python main.py
```
*Configure camera inputs, SMPL paths, and parameters in [`configs/pipeline.yml`](configs/pipeline.yml).*

### 2. Multi-View Combinatorial Benchmark (Brute-Force)
Evaluate all pairwise camera permutations across sequence segments:
```bash
python brute_force_runner.py
```
Outputs comprehensive performance metrics to `output/brute_force_report.csv` and `output/brute_force_report.html`, ranked by $\% \Delta \text{PA-MPJPE}$.

### 3. Decision Tree Factor Analysis
Extract dominant decision rules and feature importances governing fusion gains:
```bash
python analyze_tree.py
```

### 4. Regression & Verification Suite
Run continuous verification to ensure mathematical consistency and zero-GT isolation:
```bash
pytest tests/
python test_anchors_experiment.py
```

---

## 📂 Repository Structure

```text
optimization_model_monocular_3/
├── configs/                   # Pipeline configuration and keypoint schemas
├── fusion_pipeline/           # Core library
│   ├── detector.py            # 3D ray-casting occlusion detection
│   ├── correction.py          # Kinematic bone rotation & RANSAC alignment
│   ├── optimization.py        # 3D constrained optimizer with anti-negative transfer
│   └── executor.py            # Multi-stage execution engine
├── _learnable_backend/        # NetBody25 neural architecture & SMPL model integration
├── input/                     # Benchmark datasets (PKL features, calibration, videos)
├── models/                    # SMPL neutral body models & joint regressors
├── output/                    # Generated predictions, evaluations, and reports
├── tests/                     # Regression and zero-GT compliance tests
├── main.py                    # Dual-view pipeline entry point
├── brute_force_runner.py      # Exhaustive multi-view benchmark runner
├── analyze_tree.py            # Decision tree interpretable analysis script
├── MODEL_CHANGES.md           # Mathematical formulations and ablation audit
└── compat.py                  # Runtime backward compatibility layer
```

---

## 📝 Citation

If you find this code or method useful in your research, please cite:

```bibtex
@article{hieu2026robustposefusion,
  title={Robust Multi-View 3D Human Pose Fusion via Ray-Casting Occlusion Awareness and Kinematic Anti-Negative Transfer},
  author={Doan, Trung Hieu and Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
