# AI-Based Cataract Clinical Decision Support System (CDSS)
## College Presentation, Project Report & Viva Defense Guide

---

### 1. Final Project Statement (Memorize for Viva & Presentations)
> *"Our project is an AI-based Clinical Decision Support System for cataract monitoring. It analyzes eye images to detect and assess cataracts, compares images across patient visits, monitors possible progression, assesses risk, recommends an approximate follow-up period, and generates a structured report for ophthalmologists. After reviewing the AI-assisted results, the doctor makes the final clinical decision and can provide the patient with an approved check-up summary."*

---

### 2. The 14 Project Modules Explained in Plain English

| # | Module Name | Simple Explanation |
| :--- | :--- | :--- |
| **1** | **Patient Management** | Records patient ID, age, gender, monitored eye (OD/OS), and risk factors (e.g. diabetes). |
| **2** | **Image Ingestion** | Uploads and securely stores slit-lamp eye photos linked to the patient's visit. |
| **3** | **Image Preprocessing** | Uses OpenCV to check for blur, enhance contrast (CLAHE), and standardize image size. |
| **4** | **Cataract Detection** | Deep Learning CNN model identifies if the image is `Normal` or `Cataract Detected`. |
| **5** | **Severity Classification** | Evaluates optical cloudiness to categorize `Early`, `Moderate`, or `Advanced` stages. |
| **6** | **Image Comparison** | Compares baseline vs. current visit images side-by-side using SSIM (Structural Similarity). |
| **7** | **Progression Risk** | Calculates composite risk ($R = 0.45 \cdot S + 0.35 \cdot D + 0.20 \cdot C$) into `Low`, `Medium`, `High`. |
| **8** | **Follow-up Recommendation** | Recommends target month (e.g., *"Recommended follow-up: November 2026 (~3 months)"*). |
| **9** | **Patient History Log** | Keeps a chronological multi-visit timeline for each patient. |
| **10** | **Progression Trajectory Graph** | Visualizes severity & optical density changes over time on a clean dashboard graph. |
| **11** | **Explainable AI (Grad-CAM)** | Overlays visual heatmaps showing the exact lens area that influenced the AI prediction. |
| **12** | **Doctor Dashboard** | A unified doctor web interface with side-by-side comparison, Grad-CAM toggle, and notes. |
| **13** | **Clinical PDF Report** | Automatically generates an official medical document with doctor sign-off and disclaimers. |
| **14** | **Patient Check-up Summary** | Produces an easy-to-understand 1-page takeaway leaflet for the patient. |

---

### 3. Top 5 Viva Questions & Ideal Answers

#### Q1: Why is this called a "Clinical Decision Support System" and not an "Automated Diagnosis System"?
> **Answer:**  
> In healthcare, AI must act as an assistant, not an autonomous authority. Our CDSS provides objective quantitative metrics (SSIM, opacity density, Grad-CAM saliency), but the final medical diagnosis, prescription, and surgical timing remain under the legal and professional authority of the attending ophthalmologist.

#### Q2: What makes your project different from standard cataract classification projects?
> **Answer:**  
> Most student projects take a single image and simply output "Cataract or Normal". Our project performs **longitudinal multi-visit monitoring**: it compares prior baseline vs current images, measures structural changes over elapsed months, calculates a multimodal progression risk score, recommends follow-up calendar months, and provides explainable Grad-CAM heatmaps.

#### Q3: How do you prevent data leakage during deep learning training?
> **Answer:**  
> We enforce **Patient-Level Splitting**. If a patient has multiple visit images, all of their images are assigned exclusively to either the Training, Validation, or Test set. Images from the same patient never appear in both training and testing.

#### Q4: How does the system handle different lighting and cameras across visits?
> **Answer:**  
> In Module 3, we use OpenCV to apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** in the LAB color space to equalize illumination variations. Additionally, Module 6 computes structural similarity (SSIM) rather than raw Euclidean pixel distance.

#### Q5: Where does the AI model get trained?
> **Answer:**  
> We train our deep learning backbone (Transfer Learning with ResNet-50 / EfficientNet) using Google Colab or Kaggle with GPU acceleration, using public benchmark datasets like ODIR-5K and Kaggle Cataract datasets.

---

### 4. How to Run the Project on Your Computer

1. Open your terminal / command prompt.
2. Navigate to the project directory:
   ```bash
   cd C:\Users\ASUS\.gemini\antigravity-ide\scratch\cataract-cdss
   ```
3. Run the 1-click startup script:
   ```bash
   py run.py
   ```
4. Your browser will automatically open to `http://localhost:8000` with the complete Doctor Decision Support Dashboard!
