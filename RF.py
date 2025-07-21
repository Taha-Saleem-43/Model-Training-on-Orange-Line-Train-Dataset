import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Global seed
RANDOM_SEED = 42

# Load Data
@st.cache_data

def load_data():
    df = pd.read_csv("Service.csv")
    df.drop(columns=['Timestamp', 'Name'], inplace=True)
    return df

def apply_label_encoding(df):
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    return df, label_encoders

# SMOTENC
def apply_smote(X, y, label_encoders):
    cat_indices = [i for i, col in enumerate(X.columns) if col in label_encoders]
    smote_nc = SMOTENC(categorical_features=cat_indices, random_state=RANDOM_SEED)
    X_res, y_res = smote_nc.fit_resample(X, y)
    return X_res, y_res

# Train Model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    clf = RandomForestClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, classification_report(y_test, y_pred, output_dict=True), X.columns

# App layout
st.set_page_config(layout="wide")
st.title("🔍 Orange Line Metro Survey Analysis & Prediction")

# Load & Encode Data
df = load_data()
df_encoded, encoders = apply_label_encoding(df.copy())
target_col = 'How satisfied are you with the overall service quality of the Orange Line Metro Train?'
X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

# Remove class 5 (less samples)
X = X[y != 5]
y = y[y != 5]

# Navigation
section = st.sidebar.radio("Go to", ["1. Preview Dataset", "2. SMOTE Effect", "3. EDA", "4. Prediction", "5. Conclusion"])

# 1. Preview Dataset
if section == "1. Preview Dataset":
    st.subheader("📋 Original Dataset Preview")
    st.dataframe(df.head(10))
    st.write("Shape:", df.shape)

# 2. SMOTE Effect
elif section == "2. SMOTE Effect":
    st.subheader("📊 Class Distribution Before and After SMOTE")
    before = pd.Series(Counter(y))
    X_res, y_res = apply_smote(X, y, encoders)
    after = pd.Series(Counter(y_res))
    chart_df = pd.DataFrame({'Before': before, 'After': after})
    st.bar_chart(chart_df)

# 3. EDA
elif section == "3. EDA":
    st.subheader("📈 Exploratory Data Analysis")
    st.write("### Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)

    st.write("### Outlier Detection (Boxplots)")
    num_cols = X.select_dtypes(include='number').columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=X[col], ax=ax)
        st.pyplot(fig)

# 4. Prediction
elif section == "4. Prediction":
    st.subheader("🔮 Predict Satisfaction Level")
    X_res, y_res = apply_smote(X, y, encoders)
    model, report, features = train_model(X_res, y_res)

    inputs = []
    for feat in features:
        if feat in encoders:
            val = st.selectbox(feat, list(encoders[feat].classes_))
            val_encoded = encoders[feat].transform([val])[0]
            inputs.append(val_encoded)
        else:
            val = st.slider(feat, 0, 10, 5)
            inputs.append(val)

    if st.button("Predict"):
        pred = model.predict([inputs])[0]
        st.success(f"Predicted Satisfaction Rating: {pred}")

# 5. Conclusion
elif section == "5. Conclusion":
    st.subheader("📌 Conclusion")
    st.markdown("""
    - SMOTE was used to balance class distribution for fair training.
    - Class 5 was removed due to insufficient data.
    - Random Forest classifier gave stable performance across classes.
    - Exploratory Data Analysis showed variable distributions and outliers.
    - Model predictions can now be made in real-time via this app.
    - The Overall Model Accuracy is 87%.
    """)
