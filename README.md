# Project Milestone – Uncertainty-Aware 3D Spleen Segmentation with MONAI

- **Start date:** October 25  
- **Target completion date:** December 29  

## 1. Progress and Current Status

### 1.1 What has been achieved so far

- Pivoted the project from the original MURA X-ray classification idea to a new topic:
  **3D spleen segmentation with uncertainty estimation using MONAI**.
- Selected **Medical Segmentation Decathlon – Task09_Spleen** as the primary dataset.
- Studied the official MONAI 3D spleen segmentation tutorial and related documentation to
  understand the baseline pipeline (data transforms, 3D UNet, training loop).
- Defined the overall project goals:
  - Reproduce a MONAI 3D UNet spleen segmentation baseline.
  - Extend it with **Monte Carlo Dropout–based uncertainty** and **calibration analysis**.
  - Build simple visualization tools to inspect segmentation and uncertainty maps.
- Drafted a new written **project proposal** describing the updated topic, dataset, and
  evaluation plan.
- Organized the repository structure for the new project components.

> **GitHub repository:** `https://github.com/vintu2001/272_group7_project`  
> (New MONAI-related code and documentation will be added under a dedicated folder, e.g.
> `monai_spleen_uncertainty/`.)

### 1.2 Planned modules and their current status

| Module                                  | Description                                                                                      | Status               | Code link (planned) |
|-----------------------------------------|--------------------------------------------------------------------------------------------------|----------------------|---------------------|
| Dataset & I/O pipeline                  | Download Task09_Spleen, convert to a consistent structure, create MONAI `Dataset` / `DataLoader` | **In design**        | `monai_spleen_uncertainty/data/` |
| Data preprocessing & augmentation       | Spacing, intensity normalization, random cropping, flipping, etc. using MONAI dictionary transforms | **In design**        | `monai_spleen_uncertainty/transforms.py` |
| Baseline 3D UNet segmentation model     | Standard 3D UNet from MONAI with Dice + cross-entropy loss                                       | **Planned**          | `monai_spleen_uncertainty/train_baseline.py` |
| Baseline evaluation script              | Compute Dice and Hausdorff distance on validation/test sets                                      | **Planned**          | `monai_spleen_uncertainty/eval_baseline.py` |
| Uncertainty (MC Dropout) inference      | Enable dropout at inference; perform multiple forward passes to estimate voxel-wise variance     | **Planned**          | `monai_spleen_uncertainty/infer_mc_dropout.py` |
| Calibration & reliability analysis      | Compute confidence histograms, Expected Calibration Error (ECE), and reliability curves          | **Planned**          | `monai_spleen_uncertainty/calibration_analysis.py` |
| Visualization / demo                    | Jupyter notebook or simple app to visualize CT slices, segmentation masks, and uncertainty maps | **Planned**          | `monai_spleen_uncertainty/demo.ipynb` |

At this milestone, the focus has been on **topic selection, literature/tool review, and design**.  
Implementation and coding of the baseline model and modules will be done in early December to catch up.

---

## 2. Baseline Modules: Functionality and Evaluation

### 2.1 Baseline segmentation module

**Baseline model:**  
- A **3D U-Net** implemented with MONAI.  
- Input: 3D abdominal CT volume (Task09_Spleen).  
- Output: voxel-wise probability of spleen vs. background.

**Functionality:**

1. Load a preprocessed 3D CT volume and corresponding ground-truth label mask.
2. Apply MONAI transforms (e.g., `Spacingd`, `NormalizeIntensityd`, random cropping).
3. Train the 3D UNet with a Dice + cross-entropy loss function.
4. Save trained weights and log basic training statistics (loss, Dice score).

**Planned evaluation:**

- Primary metric: **Dice similarity coefficient** on validation/test sets.
- Secondary metric: **Hausdorff distance** to capture boundary quality.
- Learning curves (training/validation loss and Dice) to analyze convergence.

As of this milestone, the **design and evaluation plan** for the baseline module are defined, and we have studied the official MONAI spleen tutorial as a reference. Actual training runs and numerical evaluation will be performed once the coding phase starts.

### 2.2 Uncertainty and calibration module (extension beyond baseline)

On top of the baseline 3D UNet, we will add:

1. **Monte Carlo Dropout Inference**
   - Keep dropout layers active during inference.
   - Run multiple forward passes (e.g., 20–30) for each volume.
   - Compute the mean prediction (segmentation) and voxel-wise variance (uncertainty map).

2. **Calibration and Reliability Analysis**
   - Aggregate predicted probabilities and ground-truth labels across the test set.
   - Compute **Expected Calibration Error (ECE)**.
   - Generate **reliability diagrams** (confidence vs. accuracy).
   - Identify cases where the model is over-confident but wrong.

These modules will play a key role in the project by shifting the focus from “just segmentation accuracy” to **trustworthy predictions**, which is important in real medical applications.

---

## 3. References Used So Far

- Project MONAI – tutorials and documentation for medical image segmentation.  
- MONAI 3D Spleen Segmentation Tutorial (Task09_Spleen).  
- NVIDIA NGC MONAI Spleen CT Segmentation model card and examples.  
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
- Background materials on uncertainty estimation and calibration in deep learning (MC Dropout, ECE, reliability diagrams).

(Full citation details will be expanded in the final report.)

---

## 4. Challenges and Plans to Move Forward

### Challenges encountered

- **Learning curve for MONAI and 3D medical imaging:**
  Understanding dictionary-based transforms, 3D data handling, and typical pipelines takes time.
- **Compute and memory constraints:**
  3D segmentation models and high-resolution CT volumes can be memory-intensive.
- **Time management:**
  Pivoting away from the original MURA project has delayed the start of coding; we still need to implement and train the baseline model.

### Plans to overcome these challenges

- Follow and adapt the **official MONAI spleen tutorial** as the starting point to reduce implementation risk.
- Start with **smaller patch sizes and batch sizes** to fit into available GPU memory, then scale up if possible.
- Prioritize the following sequence for the remaining timeline:
  1. **Early December:** Implement and run the baseline MONAI 3D UNet training and evaluation.
  2. **Mid December:** Implement MC Dropout inference, generate uncertainty maps, and run calibration analysis.
  3. **Late December (before Dec 29):** Finalize visualization notebook/app, perform error analysis, and polish the report and slides.
- Schedule a face-to-face or online check-in with the instructor after the initial baseline training is complete to confirm we are on the right track.

---
