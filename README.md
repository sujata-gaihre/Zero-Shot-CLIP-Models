# Multimodal AI & Regression Analysis Projects

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.0%2B-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📂 Project Overview
This repository contains two distinct machine learning tasks demonstrating expertise in **Vision-Language Models (VLMs)** and **Predictive Regression Modeling**.

* **Task 1:** Zero-Shot Robustness Analysis & Prompt Engineering with CLIP.
* **Task 2:** California Housing Price Prediction using Ensemble Learning.

---

## 🟢 Task 1: Zero-Shot Robustness Analysis with CLIP

### 📌 Overview
This task investigates the **Zero-Shot classification capabilities** of OpenCLIP (**ViT-B-32**) on the **CIFAR-10** dataset. Unlike traditional supervised learning, this project utilizes **Prompt Engineering** to align textual and visual feature spaces without fine-tuning.

### 🚀 Key Features
* **Zero-Shot Inference:** Implementation of an OpenCLIP pipeline to classify images using natural language prompts.
* **Prompt Engineering Experiments:** Systematic evaluation of 5 distinct prompt templates to optimize semantic alignment.
* **Advanced Error Analysis:** Generation of Confusion Matrices and Radar Charts to visualize per-class weaknesses (e.g., biological vs. mechanical objects).

### 🧪 Experiments & Results
We tested varying prompt structures to determine which linguistic cues best retrieved correct visual features.

| Prompt Template | Accuracy | Insight |
| :--- | :--- | :--- |
| **"a drawing of a {}"** | **87.54%** | **Best Performance** |
| "a photo of a large {}" | 87.40% | Strong contender |
| "a photo of a {}" | 86.17% | Baseline |
| "a photo of a small {}" | 84.65% | Lowest performance |

**Key Findings:**
* **Top Performers:** Truck (1.9% error), Ship (3.1% error).
* **Challenging Classes:** Frog (27.5% error), Cat (20.5% error).
* **Misclassification:** Cats are most frequently confused with **Dogs** (129 instances).

---

## 🔵 Task 2: California Housing Price Prediction

### 📌 Overview
This task implements an end-to-end Machine Learning pipeline to predict median house values in California districts. It conducts a comparative analysis between baseline linear models and advanced ensemble methods.

### 🚀 Key Features
* **Exploratory Data Analysis (EDA):** Correlation heatmaps and histograms to understand feature distributions.
* **Data Preprocessing:** Rigorous outlier detection using the **Interquartile Range (IQR)** method on features like `AveRooms` and `Population` to improve model stability.
* **Model Comparison:** Evaluated Linear Regression, Ridge, Lasso, ElasticNet, Gradient Boosting, and Random Forest.
* **Hyperparameter Tuning:** Optimized **Random Forest Regressor** using `GridSearchCV`.

### 🧪 Experiments & Results
The analysis determined that ensemble methods significantly outperformed linear baselines.

| Model | R² Score | MAE (Mean Absolute Error) |
| :--- | :--- | :--- |
| **Random Forest (Tuned)** | **~0.806** | **0.327** |
| Gradient Boosting | 0.776 | 0.372 |
| Linear Regression | 0.576 | 0.533 |

**Key Findings:**
* **Outlier Removal:** Removing statistical outliers from `AveRooms` and `AveOccup` improved model generalization.
* **Feature Importance:** `MedianIncome` and location (Latitude/Longitude) were identified as the primary drivers of housing prices.

---

## 🛠️ Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/multimodal-regression-projects.git](https://github.com/yourusername/multimodal-regression-projects.git)
    cd multimodal-regression-projects
    ```

2.  **Install Dependencies**
    ```bash
    pip install torch torchvision open_clip_torch matplotlib numpy seaborn pandas scikit-learn
    ```

3.  **Run the Notebooks**
    * For CLIP Analysis: Run `clip_zero_shot.ipynb`
    * For Housing Prediction: Run `housing_prediction.ipynb`

## 👤 Author
**Sujata Gaihre**
*Research Interest: Multimodal AI, Zero-Shot Learning, and Predictive Modeling.*
