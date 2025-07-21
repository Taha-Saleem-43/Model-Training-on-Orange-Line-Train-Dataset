# 🚆 Orange Line Metro Train - Service Quality Prediction App

This is a **Streamlit-based Machine Learning application** that predicts customer satisfaction levels for the **Orange Line Metro Train in Lahore** using survey data.

It includes:
- Data preprocessing
- Class balancing using SMOTENC
- Model training using **RandomForest** and **XGBoost**
- Visual Exploratory Data Analysis (EDA)
- Model performance reports
- Real-time prediction based on user input

---

## 🧠 Techniques Used

### ✅ Data Preprocessing
- **Label Encoding**: For converting categorical values into numeric.
- **Missing Values Handling**: Dropped unnecessary columns like `Timestamp` and `Name`.

### ✅ Class Imbalance Handling
- **SMOTENC (Synthetic Minority Oversampling Technique for Nominal and Continuous)**: Used to generate synthetic samples while preserving categorical integrity.

### ✅ Modeling Techniques
- **RandomForestClassifier**: Ensemble model used for baseline accuracy.
- **XGBoostClassifier**: Gradient boosting classifier used for more robust, optimized predictions. (Not Well Predict)

### ✅ Evaluation Metrics
- `classification_report`: For Precision, Recall, F1-Score
- `confusion_matrix`: Plotted via Seaborn heatmap

### ✅ Visualization and UI
- `Matplotlib`, `Seaborn` for EDA visualizations
- **Streamlit** to create interactive, real-time prediction dashboard

---

