# Credit Risk Management Using Machine Learning

This project applies machine learning techniques to assess and mitigate credit risk. It predicts the likelihood of credit card defaults using **advanced data preprocessing, feature engineering, and classification models**.

## 📌 Project Overview
- **Objective:** Predict credit card default risk based on customer financial data.
- **Dataset:** Sourced from the **AmExpert 2021 CodeLab - Machine Learning Hackathon** (30,000 rows, 19 features).
- **Methods:** Data cleaning, feature engineering, class balancing (SMOTE), and model training.

## 🚀 Features & Workflow
1. **Data Preprocessing**
   - Handling missing values, outliers, and scaling numerical features.
   - Encoding categorical variables for compatibility with ML models.
   - Using **SMOTE** to balance the class distribution.

2. **Exploratory Data Analysis (EDA)**
   - Visualizing feature distributions and correlations.
   - Identifying patterns in default behavior.

3. **Model Training & Evaluation**
   - **Algorithms Used:**
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - LightGBM
     - XGBoost (**Best Model**)
   - **Performance Metrics:** Accuracy, Precision, Recall, F1-Score.

## 📊 Model Performance
| Model            | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|----------|--------|---------|
| Logistic Regression | 94.64%  | 0.80     | 0.96   | 0.86    |
| Decision Tree   | 96.63%  | 0.87     | 0.92   | 0.90    |
| Random Forest   | 96.50%  | 0.86     | 0.94   | 0.90    |
| LightGBM       | 96.68%  | 0.87     | 0.94   | 0.90    |
| **XGBoost** (Best) | **97.34%**  | **0.91**  | **0.92** | **0.91** |

## 🛠 Tech Stack
- **Programming Language:** Python 3.x
- **Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`
- **Frameworks:** Jupyter Notebook, Google Colab

## 📂 Folder Structure
```
📁 Credit-Risk-Management-ML
│── README.md
│── requirements.txt
│── data/            # Dataset files
│── notebooks/       # Jupyter Notebooks
│── src/             # Preprocessing and model training scripts
│── results/         # Model outputs
```

## 🔧 Installation & Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/rohii52/Credit-Risk-Management-ML.git
   cd Credit-Risk-Management-ML
   ```
2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook**
   ```sh
   jupyter notebook
   ```
4. **Train Models**
   ```sh
   python src/train_model.py
   ```

## 🔮 Future Enhancements
- Deploying as an **API using Flask**
- Testing **deep learning models (RNN, Bi-LSTM)**
- Developing a **dashboard using Streamlit** for real-time predictions
