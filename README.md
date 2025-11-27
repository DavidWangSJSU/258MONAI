# Project Milestone – Uncertainty-Aware 3D Spleen Segmentation with MONAI

- **Start date:** October 25  
- **Target completion date:** November 29  

---

## 1. Progress and Current Status

### 1.1 What has been achieved so far

- Pivoted the project from the original MURA X-ray classification idea to a new topic:
  **3D spleen segmentation with uncertainty estimation using MONAI**.
- Selected **Medical Segmentation Decathlon – Task09_Spleen** as the primary dataset.
- Reviewed the official MONAI 3D spleen segmentation tutorial and documentation to
  understand the baseline pipeline (data transforms, 3D UNet, training loop).
- Defined the overall project goals:
  - Reproduce a MONAI 3D UNet spleen segmentation baseline.
  - Extend it with **Monte Carlo Dropout–based uncertainty** and **calibration analysis**.
  - Build simple visualization tools to inspect segmentation and uncertainty maps.
- Drafted an updated written **project proposal** describing the new topic, dataset, and
  evaluation plan.
- Set up the base repository for this project.

> **GitHub repository:** `https://github.com/DavidWangSJSU/258MONAI`  

At this milestone the focus has been on **topic selection, background research, and design**.  
Implementation and training of the baseline model are scheduled for early–mid November.

---

### 1.2 Planned modules and their current status

| Module                                  | Description                                                                                      | Status               | Planned Code Location |
|-----------------------------------------|--------------------------------------------------------------------------------------------------|----------------------|-----------------------|
| Dataset & I/O pipeline                  | Download Task09_Spleen, organize files, create MONAI `Dataset` / `DataLoader`                   | **Planned**          | `data/`, `dataset.py` |
| Data preprocessing & augmentation       | Spacing, intensity normalization, random cropping, flipping, etc. using MONAI transforms        | **Planned**          | `transforms.py`       |
| Baseline 3D UNet segmentation model     | Standard 3D UNet from MONAI with Dice + cross-entropy loss                                      | **Planned**          | `train_baseline.py`   |
| Baseline evaluation script              | Compute Dice and Hausdorff distance on validation/test sets                                     | **Planned**          | `eval_baseline.py`    |
| Uncertainty (MC Dropout) inference      | Enable dropout at inference; multiple forward passes; compute voxel-wise variance               | **Planned**          | `infer_mc_dropout.py` |
| Calibration & reliability analysis      | Compute confidence histograms, Expected Calibration Error (ECE), reliability curves             | **Planned**          | `calibration_analysis.py` |
| Visualization / demo                    | Jupyter notebook or simple app to visualize CT slices, masks, and uncertainty maps             | **Planned**          | `demo.ipynb`          |

---

### 1.3 Timeline (with dates: work items and status)

**October 25**  
- Project officially starts with the original medical-imaging idea.

**Oct 25 – Nov 1**  
- Explore initial project direction (MURA-based classification).   
- Research **Project MONAI** and Task09_Spleen dataset.  
- Decide to pivot to **uncertainty-aware 3D spleen segmentation with MONAI**.  
- Set up base repository `258MONAI`.  
- Draft and finalize the updated project proposal.  
- **Status:** Completed / in progress.

**Nov 2 – Nov 8**  
- Implement dataset download and organization scripts in the repository.  
- Implement MONAI transforms and `Dataset`/`DataLoader` for 3D spleen segmentation.  
- Verify that the pipeline can load and iterate through volumes without training yet.  
- **Status:** Planned.

**Nov 9 – Nov 15**  
- Implement baseline 3D UNet training script (`train_baseline.py`).  
- Run initial training on a subset of the data to debug the pipeline.  
- Run full training on the training set with tuned hyperparameters.  
- Implement evaluation script (`eval_baseline.py`) and compute Dice / Hausdorff on test set.  
- Save trained model checkpoints and record baseline results.  
- **Status:** Completed.

**Nov 16 – Nov 22**  
- Implement MC Dropout inference (`infer_mc_dropout.py`) to generate uncertainty maps.  
- Implement calibration analysis (`calibration_analysis.py`) with ECE and reliability diagrams.  
- Generate initial uncertainty visualizations on selected test cases.  
- **Status:** Completed.

**Nov 23 – Nov 27**  
- Create visualization notebook (`demo.ipynb`) to show CT slices, segmentation, and uncertainty.  
- Perform error analysis and select representative examples (good, borderline, and failure cases).  
- Draft main sections of the final report (Methods, Experiments, Results, Discussion).  
- **Status:** In Progress.

**Nov 28 – Nov 29**  
- Finalize report, README, and comments in the `258MONAI` repository.  
- Prepare any slides or summary documents required by the course.  
- Submit project materials and complete project.  
- **Status:** In Porgress.

---

## 2. Baseline Modules: Functionality and Evaluation

### 2.1 Baseline segmentation module

**Baseline model:**  
- A **3D U-Net** implemented with MONAI.  
- Input: 3D abdominal CT volume (Task09_Spleen).  
- Output: voxel-wise probability of spleen vs. background.

**Functionality (planned):**

1. Load a preprocessed 3D CT volume and corresponding ground-truth mask.
2. Apply MONAI transforms (spacing, normalization, random cropping, etc.).
3. Train the 3D UNet with a Dice + cross-entropy loss function.
4. Save trained weights and log training statistics (loss, Dice).

**Planned evaluation:**

- Primary metric: **Dice similarity coefficient** on validation/test sets.  
- Secondary metric: **Hausdorff distance** for boundary quality.  
- Learning curves to analyze convergence and overfitting.

At this milestone, the **design and evaluation plan** are defined; implementation and training are scheduled for November.

### 2.2 Uncertainty and calibration module

On top of the baseline 3D UNet, we will add:

1. **Monte Carlo Dropout Inference**
   - Keep dropout layers active during inference.
   - Run multiple forward passes for each volume.
   - Compute mean prediction and voxel-wise variance (uncertainty map).

2. **Calibration and Reliability Analysis**
   - Aggregate predicted probabilities and ground-truth labels across the test set.
   - Compute **Expected Calibration Error (ECE)**.  
   - Generate **reliability diagrams** (confidence vs. accuracy).  
   - Identify cases where the model is confidently wrong.

These extensions shift the project from “just segmentation accuracy” to **trustworthy prediction**, which is important in real medical settings.

---

## 3. References Used So Far

- Project MONAI – tutorials and documentation for medical image segmentation.  
- MONAI 3D Spleen Segmentation Tutorial (Task09_Spleen).  
- NVIDIA NGC MONAI Spleen CT Segmentation model card.  
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.*  
- Background readings on Monte Carlo Dropout, uncertainty estimation, and calibration (ECE, reliability diagrams).

(Full citation details will be expanded in the final report.)

---

