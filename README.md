# 🚀 Codveda Machine Learning Internship Portfolio

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20(100%25)-brightgreen)](#)
[![Internship](https://img.shields.io/badge/Internship-Codveda%20Technologies-6C63FF)](#)

> **Author:** [Pijus Saha](https://github.com/Pijus-Saha)  
> **Program:** Machine Learning Internship at **Codveda Technologies**  
> **Repository:** [codveda-ml-internship](https://github.com/Pijus-Saha/codveda-ml-internship)

---

## 📌 Executive Summary

This repository houses my selected task submissions for the **Machine Learning Internship at Codveda Technologies**. The curriculum spans three progressive proficiency tiers—**Level 1 (Basic)**, **Level 2 (Intermediate)**, and **Level 3 (Advanced)**. 

To satisfy the internship graduation requirements, **two targeted machine learning tasks** are selected and implemented per level (totaling 6 comprehensive task implementations). Each task notebook adheres to industry-grade data science practices:
- **Zero data leakage**: Strict train/test isolation for scalers, imputers, and transformers.
- **Robust exploratory data analysis (EDA)**: Understanding data distributions, missing values, and collinearity.
- **Systematic hyperparameter optimization**: Cost-complexity pruning paths, elbow method inertia curves, cross-validated randomized search, and neighborhood evaluation curves.
- **Rigorous evaluation**: Multi-metric evaluation including Stratified Accuracy, Precision, Recall, F1-score, Confusion Matrices, Cluster Profiles, Feature Importance rankings, and Learning Curves.

---

## 📂 Repository Architecture

```text
codveda-ml-internship/
├── datasets/                                # Raw and structured project datasets
│   ├── churn-bigml-80.csv                  # Telecom customer churn (80% training set - 2,666 rows)
│   ├── churn-bigml-20.csv                  # Telecom customer churn (20% evaluation set - 667 rows)
│   ├── iris.csv                            # Classic Fisher's Iris taxonomy dataset (150 samples)
│   ├── house_Prediction_Data Set.csv       # Real estate continuous target regression dataset
│   ├── Sentiment_dataset.csv               # Textual sentiment classification dataset
│   ├── Stock_Prices_Data_Set.csv           # Financial time-series pricing dataset
│   └── Machine Learning Task List.pdf      # Official Codveda task guidelines & syllabus
│
├── Level_1_Basic/                           # Foundational Machine Learning Pipelines
│   ├── task1_preprocessing.ipynb           # Task 1: End-to-end data preprocessing & encoding
│   └── task3_knn_classifier.ipynb          # Task 3: K-Nearest Neighbors multi-class classification
│
├── Level_2_Intermediate/                    # Intermediate Supervised & Unsupervised Modeling
│   ├── task2_decision_tree.ipynb           # Task 2: Decision Tree classification & post-pruning
│   └── task3_kmeans_clustering.ipynb       # Task 3: K-Means clustering & customer segmentation
│
├── Level_3_Advanced/                        # Advanced Ensembles & Deep Learning Architectures
│   ├── task1_random_forest.ipynb           # Task 1: Tuned Random Forest ensemble & feature importance
│   └── task3_neural_network.ipynb          # Task 3: Feed-forward deep neural network (MLP)
│
└── README.md                                # Project documentation & technical walkthrough
```

---

## 🔬 Detailed Task Implementations

### 🟢 Level 1: Basic Machine Learning

#### 🔹 [Task 1: Data Preprocessing for Machine Learning](file:///e:/codveda-ml-internship/Level_1_Basic/task1_preprocessing.ipynb)
* **Status:** `Completed` ✅
* **Notebook:** [`Level_1_Basic/task1_preprocessing.ipynb`](Level_1_Basic/task1_preprocessing.ipynb)
* **Dataset:** Telecom Churn (`churn-bigml-80.csv`, 2,666 entries, 20 features)
* **Core Objective:** Build a production-grade, defensive preprocessing pipeline transforming raw heterogeneous telecom customer data into scaled numeric arrays.
* **Pipeline Methodology:**
  1. **Data Inspection & Integrity:** Analyzed data types, feature distributions, and null values across 16 numerical and 3 categorical features.
  2. **Defensive Imputation:** Implemented `SimpleImputer` using **median** strategy for numerical columns (outlier-resistant telecom call minutes) and **most frequent (mode)** strategy for categorical features.
  3. **Categorical Encoding:** Applied binary mapping (`Yes`/`No` $\to$ `1`/`0`) for binary flags (`International plan`, `Voice mail plan`) and One-Hot Encoding for nominal variables.
  4. **Stratified Splitting:** Partitioned the dataset using an 80/20 train-test split (`random_state=42`) with target stratification on `Churn` to preserve critical class imbalance (2,132 training samples, 534 test samples).
  5. **Data-Leakage Free Scaling:** Fitted `StandardScaler` strictly on the training set continuous features and transformed the test partition accordingly.

---

#### 🔹 [Task 3: K-Nearest Neighbors (KNN) Classifier](file:///e:/codveda-ml-internship/Level_1_Basic/task3_knn_classifier.ipynb)
* **Status:** `Completed` ✅
* **Notebook:** [`Level_1_Basic/task3_knn_classifier.ipynb`](Level_1_Basic/task3_knn_classifier.ipynb)
* **Dataset:** Iris Flower Dataset (`iris.csv`, 150 samples, 4 continuous features, 3 species)
* **Core Objective:** Implement a K-Nearest Neighbors classifier, explore the bias-variance tradeoff across varying neighborhood sizes ($K$), and evaluate multi-class discrimination.
* **Pipeline Methodology:**
  1. **Train/Test Stratification:** 80/20 stratified split (120 training samples, 30 test samples).
  2. **Feature Normalization:** Standardized all four morphological dimensions (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) using `StandardScaler` to ensure Euclidean distance parity.
  3. **Hyperparameter Search:** Evaluated model performance across neighborhood sizes $K \in [1, 20]$, tracking training accuracy versus generalization test accuracy.
  4. **Optimal Neighborhood Selection & Evaluation:** Selected optimal $K$ resolving tie-breaks on generalization performance.
* **Performance Results:**
  * **Test Accuracy:** **96.67%** (29/30 correct classifications)
  * **Per-Class Metrics:**
    * `Setosa`: Precision = **1.00**, Recall = **1.00**, F1-Score = **1.00**
    * `Versicolor`: Precision = **0.91**, Recall = **1.00**, F1-Score = **0.95**
    * `Virginica`: Precision = **1.00**, Recall = **0.90**, F1-Score = **0.95**
  * **Visualizations:** Training vs. Test Accuracy curve across $K$, Annotated Confusion Matrix Heatmap.

---

### 🟡 Level 2: Intermediate Machine Learning

#### 🔹 [Task 2: Decision Trees for Classification & Pruning](file:///e:/codveda-ml-internship/Level_2_Intermediate/task2_decision_tree.ipynb)
* **Status:** `Completed` ✅
* **Notebook:** [`Level_2_Intermediate/task2_decision_tree.ipynb`](Level_2_Intermediate/task2_decision_tree.ipynb)
* **Dataset:** Iris Flower Dataset (`iris.csv`, 150 samples)
* **Core Objective:** Train an interpretable Decision Tree classifier, diagnose tree complexity, and implement **Minimal Cost-Complexity Pruning (CCP)** to prevent overfitting.
* **Pipeline Methodology:**
  1. **Initial Unpruned Tree Analysis:** Extracted the pruning path (`cost_complexity_pruning_path`) calculating effective alphas ($\alpha$) and impurity profiles.
  2. **Alpha Regularization Tuning:** Iterated through candidate `ccp_alpha` values, fitting decision trees and analyzing test set generalization versus tree depth and leaf counts.
  3. **Optimal Model Extraction:** Selected optimal parameter `ccp_alpha = 0.0063`, yielding a streamlined, non-overfitted tree architecture.
  4. **Tree Structure Visualization:** Rendered high-resolution decision tree diagrams depicting split thresholds, Gini impurity values, sample counts, and dominant class labels.
* **Performance Results:**
  * **Test Accuracy:** **96.67%**
  * **Weighted F1-Score:** **96.66%**
  * **Confusion Matrix:** Flawless classification for *Setosa* (10/10), 90% recall on *Versicolor*, 100% precision on *Virginica*.

---

#### 🔹 [Task 3: K-Means Clustering for Customer Segmentation](file:///e:/codveda-ml-internship/Level_2_Intermediate/task3_kmeans_clustering.ipynb)
* **Status:** `Completed` ✅
* **Notebook:** [`Level_2_Intermediate/task3_kmeans_clustering.ipynb`](Level_2_Intermediate/task3_kmeans_clustering.ipynb)
* **Dataset:** Telecom Churn (`churn-bigml-80.csv`, 2,666 entries)
* **Core Objective:** Perform unsupervised customer behavioral segmentation using K-Means clustering and dimensionality reduction for visual inspection.
* **Pipeline Methodology:**
  1. **Behavioral Feature Subsetting:** Isolated continuous call consumption features (day, evening, night, international minutes and charges, customer service call frequency, voicemail volume).
  2. **Standardization:** Zero-mean, unit-variance scaling via `StandardScaler` to prevent high-magnitude features (e.g., call minutes) from skewing inertia distances.
  3. **Elbow Method:** Evaluated Within-Cluster Sum of Squares (WCSS / Inertia) across $K \in [1, 10]$ with 10 random initializations each (`n_init=10`), identifying the elbow bend at **$K = 3$**.
  4. **Dimensionality Reduction (PCA):** Projected multi-dimensional feature space onto 2 principal components to visualize cluster boundaries and centroid separations.
  5. **Cluster Profiling:** Aggregated and interpreted customer usage personas:
     * **Cluster 0 (Moderate/Standard Users):** Average usage patterns across all times of day, moderate service call volume.
     * **Cluster 1 (Night & Off-Peak Heavy):** Disproportionate off-peak evening/night call duration.
     * **Cluster 2 (High-Value Peak Spenders):** High daytime minutes, high international activity, high revenue drivers.

---

### 🔴 Level 3: Advanced Machine Learning

#### 🔹 [Task 1: Build a Random Forest Classifier](file:///e:/codveda-ml-internship/Level_3_Advanced/task1_random_forest.ipynb)
* **Status:** `Completed` ✅
* **Notebook:** [`Level_3_Advanced/task1_random_forest.ipynb`](Level_3_Advanced/task1_random_forest.ipynb)
* **Dataset:** Telecom Churn (`churn-bigml-80.csv`, 2,666 entries, 20 features)
* **Core Objective:** Build, tune, and evaluate an ensemble Random Forest classifier on structured customer churn data and extract Mean Decrease in Impurity (Gini importance) feature attributions.
* **Pipeline Methodology:**
  1. **Feature Engineering & One-Hot Encoding:** Binary-mapped service plans (`International plan`, `Voice mail plan`) and applied One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) to categorical features, creating a 68-dimensional feature matrix.
  2. **Stratified Partitioning:** Executed an 80/20 train/test split with target stratification on `Churn` (2,132 training samples, 534 holdout test samples).
  3. **Hyperparameter Optimization via RandomizedSearchCV:** Configured `RandomizedSearchCV` with 3-fold cross-validation across 20 candidate combinations covering:
     * `n_estimators`: `[100, 200, 300, 400]`
     * `max_depth`: `[10, 20, 30, None]`
     * `min_samples_split`: `[2, 5, 10]`
     * `min_samples_leaf`: `[1, 2, 4]`
     * `bootstrap`: `[True, False]`
  4. **Optimal Parameter Identification:** Extracted best hyperparameter profile:
     `{'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30, 'bootstrap': False}`.
  5. **Cross-Validation Verification:** Performed 5-fold cross-validation on the training partition yielding an accuracy of **93.95% (+/- 0.0035)**.
  6. **Feature Importance Attribution:** Calculated Gini importance scores and generated a barplot ranking the top 15 churn drivers, confirming that call charges/minutes (`Total day charge`, `Total day minutes`, `Customer service calls`, `International plan_Yes`, `Total eve charge`) dominate customer defection probability.
* **Performance Results:**
  * **Test Set Accuracy:** **94.76%** (506/534 correct classifications)
  * **5-Fold Cross-Validation Accuracy:** **93.95%**
  * **Per-Class Metrics:**
    * `Non-Churn (0)`: Precision = **0.94**, Recall = **1.00**, F1-Score = **0.97** (Support: 456)
    * `Churn (1)`: Precision = **1.00**, Recall = **0.64**, F1-Score = **0.78** (Support: 78)
  * **Macro Avg:** Precision = **0.97**, Recall = **0.82**, F1-Score = **0.88**
  * **Weighted Avg:** Precision = **0.95**, Recall = **0.95**, F1-Score = **0.94**
  * **Visualizations:** High-resolution horizontal barplot of the Top 15 Feature Importances.

---

#### 🔹 [Task 3: Deep Neural Networks with TensorFlow / Keras](file:///e:/codveda-ml-internship/Level_3_Advanced/task3_neural_network.ipynb)
* **Status:** `Completed` ✅
* **Notebook:** [`Level_3_Advanced/task3_neural_network.ipynb`](Level_3_Advanced/task3_neural_network.ipynb)
* **Dataset:** Telecom Churn (`churn-bigml-80.csv`, 2,666 entries)
* **Core Objective:** Design, train, and diagnose a Deep Feed-Forward Neural Network (Multi-Layer Perceptron) for binary customer churn classification using TensorFlow / Keras with dropout regularization and early stopping.
* **Pipeline Methodology:**
  1. **Strict Input Standardization:** One-hot encoded categorical variables with `drop_first=True` (68 features). Standardized all features using `StandardScaler` fitted strictly on training data to ensure stable gradient descent and prevent gradient saturation.
  2. **Deep MLP Architecture:** Constructed a modular `Sequential` deep neural network:
     * **Input Layer:** 68 input dimensions matching preprocessed features.
     * **Dense Layer 1:** 64 hidden units with **ReLU** activation (4,416 parameters).
     * **Dropout Regularization 1:** 30% dropout rate (`Dropout(0.3)`) to prevent node co-adaptation.
     * **Dense Layer 2:** 32 hidden units with **ReLU** activation (2,080 parameters).
     * **Dropout Regularization 2:** 20% dropout rate (`Dropout(0.2)`).
     * **Output Layer:** 1 neuron with **Sigmoid** activation for calibrated churn probabilities (33 parameters).
     * **Total Trainable Parameters:** 6,529.
  3. **Compilation & Optimization:** Compiled using the **Adam** adaptive learning rate optimizer and **Binary Crossentropy** loss function.
  4. **Training Dynamics & Early Stopping:**
     * Max Epochs: 100 with batch size of 32.
     * Internal Validation: 20% validation split on training data (427 validation samples).
     * Callback: `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`.
     * Convergence: Early stopping triggered at **Epoch 30**, restoring best weights to prevent overfitting.
  5. **Diagnostic Visualizations:** Generated 1x2 dual subplots showing Model Accuracy and Model Loss curves across all training epochs, confirming convergence and generalization.
* **Performance Results:**
  * **Test Set Accuracy:** **85.96%** (on completely unseen holdout set)
  * **Test Set Loss:** **0.4187** (Binary Cross-Entropy)
  * **Training Convergence:** Train Accuracy = **94.02%**, Validation Accuracy = **87.12%**
  * **Visualizations:** Dual Training vs. Validation Learning Curves (Accuracy & Loss over Epochs).

---

## 📊 Performance Benchmarks Summary

| Level | Task | Algorithm | Dataset | Key Metric / Result | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Level 1** | **Task 1** | Data Preprocessing Pipeline | Telecom Churn | Zero leakage, 80/20 Stratified Split, Scaled | ✅ Completed |
| **Level 1** | **Task 3** | K-Nearest Neighbors (KNN) | Iris Dataset | **Accuracy: 96.67%** \| F1: 0.97 | ✅ Completed |
| **Level 2** | **Task 2** | Pruned Decision Tree | Iris Dataset | **Accuracy: 96.67%** \| F1: 0.9666 (CCP $\alpha=0.0063$) | ✅ Completed |
| **Level 2** | **Task 3** | K-Means Clustering | Telecom Churn | **Optimal $K = 3$** (Elbow Method & PCA Segmentation) | ✅ Completed |
| **Level 3** | **Task 1** | Tuned Random Forest | Telecom Churn | **Accuracy: 94.76%** \| 5-Fold CV: 93.95% \| F1: 0.94 | ✅ Completed |
| **Level 3** | **Task 3** | Deep Neural Network (MLP) | Telecom Churn | **Accuracy: 85.96%** \| Loss: 0.4187 (Early Stopping) | ✅ Completed |

---

## 📦 Datasets Reference

| Dataset File | Domain | Rows / Columns | Primary Use Case in Repo |
| :--- | :--- | :--- | :--- |
| `churn-bigml-80.csv` | Telecommunications | 2,666 × 20 | Level 1 Task 1 (Prep), Level 2 Task 3 (K-Means), Level 3 Task 1 (RF), Level 3 Task 3 (NN) |
| `churn-bigml-20.csv` | Telecommunications | 667 × 20 | Out-of-sample holdout test partition for churn models |
| `iris.csv` | Botany / Taxonomy | 150 × 5 | Level 1 Task 3 (KNN) & Level 2 Task 2 (Decision Trees) |
| `house_Prediction_Data Set.csv` | Real Estate | 545 × 13 | Continuous variable regression analysis |
| `Sentiment_dataset.csv` | Natural Language Processing | Text Corpus | Text categorization & NLP classification |
| `Stock_Prices_Data_Set.csv` | Quantitative Finance | Time-Series | Sequential trend analysis & forecasting |

---

## ⚙️ Installation & Quickstart

Follow these instructions to reproduce the notebooks locally:

### 1. Clone the Repository
```bash
git clone https://github.com/Pijus-Saha/codveda-ml-internship.git
cd codveda-ml-internship
```

### 2. Set Up a Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab tensorflow
```

### 4. Launch Jupyter Lab
```bash
jupyter lab
```
Navigate to any notebook under `Level_1_Basic/`, `Level_2_Intermediate/`, or `Level_3_Advanced/` to inspect and run the code cells.

---

## 🛠️ Technology Stack & Libraries

* **Core Language:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn`
* **Deep Learning Framework:** `tensorflow` / `keras`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Development Environment:** Jupyter Notebook / JupyterLab, Visual Studio Code

---

## 👨‍💻 Author & Connect

**Pijus Saha**  
* **GitHub:** [@Pijus-Saha](https://github.com/Pijus-Saha)  
* **Email:** [pijussahaofficial@gmail.com](mailto:pijussahaofficial@gmail.com)
* **Website:** [pijus-saha.vercel.app](https://pijus-saha.vercel.app/)

---

## 🏷️ Tags & Acknowledgments

Special thanks to **Codveda Technologies** for providing this structured internship opportunity.

`#Codveda` `#CodvedaJourney` `#CodvedaExperience` `#FutureWithCodveda` `#MachineLearning` `#DataScience` `#Python` `#DeepLearning` `#ArtificialIntelligence`

