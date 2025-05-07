import streamlit as st  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import warnings
import pickle

# Filter warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_icon="🌐", page_title="Employee Attrition Analysis", layout="wide")
st.title(''':orange[***Welcome to the Employee Attrition Analysis system!***]''')

# Upload dataset (CSV or Pickle)
uploaded_file = st.file_uploader("Upload a CSV or Pickle file", type=["csv", "pkl"])

df = None
rf = None
model_features = None

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.pkl'):
        loaded_obj = pd.read_pickle(uploaded_file)
        if isinstance(loaded_obj, pd.DataFrame):
            df = loaded_obj
        elif isinstance(loaded_obj, tuple) and len(loaded_obj) == 2:
            rf, model_features = loaded_obj
        else:
            st.error("Invalid Pickle file. It must be a DataFrame or a (model, features) tuple.")

# Set background image (optional)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}   
[data-testid="stSidebarContent"]{
background-color: rgba(0, 0, 0, 0);
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    selected_page = option_menu(
        'Employee Attrition Analysis and Prediction', 
        ['Home', 'Predict Attrition', 'Age vs Employee-Attrition', 'Age Distribution', 'Department vs Employee-Attrition', 'Attrition Analysis'],
        icons=['house', 'activity', 'bar-chart', 'person', 'briefcase', 'pie-chart'], 
        default_index=0,
        orientation='vertical',  
        styles={
            "container": {"padding": "3px", "background-color": "#f0f0f5"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "3px", "--hover-color": "#f0f0f5"},
            "nav-link-selected": {"background-color": "#ADD8E6"},
        }
    )

# Home Page
if selected_page == 'Home':
    if df is not None:
        st.write("✅ DataFrame loaded successfully!")
        st.write(df.head())
    elif rf is not None:
        st.write("✅ Model loaded successfully and ready for prediction.")
    else:
        st.write("Please upload a CSV file for analysis or a Pickle model for prediction.")

elif selected_page == "Predict Attrition":
    if rf is not None and model_features is not None:
        st.header("Predict Employee Attrition")
        st.markdown("### 📋 Enter Employee Details")

        # Collect inputs
        age = st.slider("Age", 18, 60, 30)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)

        overtime = st.selectbox("OverTime", ["Yes", "No"])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", 
            "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        performance_rating = st.selectbox("Performance Rating", [1, 2, 3, 4])

        # Build input DataFrame
        input_dict = {
            'Age': [age],
            'JobSatisfaction': [job_satisfaction],
            'YearsAtCompany': [years_at_company],
            'MonthlyIncome': [monthly_income],
            'OverTime': [overtime],  # ✅ keep as string
            'Department': [department],
            'JobRole': [job_role],
            'MaritalStatus': [marital_status],
            'PerformanceRating': [performance_rating],
        }

        input_df = pd.DataFrame(input_dict)

        # One-hot encode categorical columns
        categorical_cols = ['Department', 'JobRole', 'MaritalStatus', 'OverTime']
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)  # ✅ match training

        # Align input with model features
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_features]

        # Predict
        prediction = rf.predict(input_encoded)[0]
        proba = rf.predict_proba(input_encoded)[0][1]

        # Display result
        st.subheader("📊 Prediction Result:")
        st.write(f"**Attrition:** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Probability of Leaving:** {proba:.2%}")
    else:
        st.warning("Please upload a Pickle model file (with both model and features) to use the prediction feature.")

# Age vs Employee-Attrition
elif selected_page == 'Age vs Employee-Attrition':
    if df is not None:
        st.title("Age vs Employee-Attrition Analysis")
        if df['Attrition'].dtype == 'object':
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        fig, ax = plt.subplots()
        ax.scatter(df['Age'], df['Attrition'], color='Slateblue', alpha=0.5)
        ax.set_xlabel('Age')
        ax.set_ylabel('Attrition')
        ax.set_title('Age vs Employee-Attrition')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to view this chart.")

# Age Distribution
elif selected_page == 'Age Distribution':
    if df is not None:
        st.title("Age Distribution Analysis")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['Age'], bins=15, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Age Distribution of Employees')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to view this chart.")

# Department vs Employee-Attrition
elif selected_page == 'Department vs Employee-Attrition':
    if df is not None:
        st.title("Department vs Employee-Attrition Analysis")
        if df['Attrition'].dtype == 'object':
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Department', y='Attrition', hue='PerformanceRating', data=df, palette='Set2')
        plt.title("Department vs Employee-Attrition")
        plt.xlabel("Department")
        plt.ylabel("Attrition")
        st.pyplot(plt)
    else:
        st.write("Please upload a CSV file to view this chart.")

# Attrition Pie Chart
elif selected_page == 'Attrition Analysis':
    if df is not None:
        st.title("Attrition Analysis")
        if df['Attrition'].dtype != 'int':
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        attrition_counts = df['Attrition'].value_counts()
        labels = ['No', 'Yes']
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(attrition_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
        ax.set_title('Attrition Distribution')
        st.pyplot(fig)
    else:
        st.write("Please upload a CSV file to view this chart.")
