import streamlit as st
import pandas as pd
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    with open('final_model_xgb_tuned_FIX_BGT_20251105_0911.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('line_explainer.dill', 'rb') as f:
        explainer = dill.load(f)
    return model, explainer

try:
    model, lime_explainer = load_model_and_explainer()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model or explainer: {e}")
    model_loaded = False

# Title
st.title("🚗 Insurance Fraud Detection System")
st.markdown("---")

# Create input form in the center
if model_loaded:
    # Create columns for centering
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.header("Enter Claim Information")
        
        # Personal Information
        st.subheader("👤 Personal Information")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            Sex = st.selectbox("Sex", ['Female', 'Male'])
        with col_b:
            MaritalStatus = st.selectbox("Marital Status", ['Single', 'Married', 'Widow', 'Divorced'])
        with col_c:
            AgeOfPolicyHolder = st.selectbox("Age of Policy Holder", 
                ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', 
                 '36 to 40', '41 to 50', '51 to 65', 'over 65'])
        
        st.markdown("---")
        
        # Vehicle Information
        st.subheader("🚙 Vehicle Information")
        col_a, col_b = st.columns(2)
        with col_a:
            Make = st.selectbox("Make", 
                ['Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac', 'Accura', 
                 'Dodge', 'Mercury', 'Jaguar', 'Nisson', 'VW', 'Saab', 'Saturn', 
                 'Porche', 'BMW', 'Mecedes', 'Ferrari', 'Lexus'])
            VehicleCategory = st.selectbox("Vehicle Category", ['Sport', 'Utility', 'Sedan'])
        with col_b:
            VehiclePrice = st.selectbox("Vehicle Price", 
                ['less than 20000', '20000 to 29000', '30000 to 39000', 
                 '40000 to 59000', '60000 to 69000', 'more than 69000'])
            AgeOfVehicle = st.selectbox("Age of Vehicle", 
                ['new', '2 years', '3 years', '4 years', '5 years', 
                 '6 years', '7 years', 'more than 7'])
        
        st.markdown("---")
        
        # Policy Information
        st.subheader("📋 Policy Information")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            PolicyType = st.selectbox("Policy Type", 
                ['Sport - Liability', 'Sport - Collision', 'Sedan - Liability', 
                 'Utility - All Perils', 'Sedan - All Perils', 'Sedan - Collision', 
                 'Utility - Collision', 'Utility - Liability', 'Sport - All Perils'])
            BasePolicy = st.selectbox("Base Policy", ['Liability', 'Collision', 'All Perils'])
            PastNumberOfClaims = st.selectbox("Past Number of Claims", 
                ['none', '1', '2 to 4', 'more than 4'])
        with col_b:
            NumberOfSuppliments = st.selectbox("Number of Supplements", 
                ['none', '1 to 2', '3 to 5', 'more than 5'])
            Days_Policy_Accident = st.selectbox("Days: Policy to Accident", 
                ['none', '1 to 7', '8 to 15', '15 to 30', 'more than 30'])
            Days_Policy_Claim = st.selectbox("Days: Policy to Claim", 
                ['none', '8 to 15', '15 to 30', 'more than 30'])
        with col_c:
            AddressChange_Claim = st.selectbox("Address Change (Claim)", 
                ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'])
            NumberOfCars = st.selectbox("Number of Cars", 
                ['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'])
        
        st.markdown("---")
        
        # Accident Details
        st.subheader("🚨 Accident Details")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            Month = st.selectbox("Month", 
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            DayOfWeek = st.selectbox("Day of Week", 
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        with col_b:
            AccidentArea = st.selectbox("Accident Area", ['Urban', 'Rural'])
            Fault = st.selectbox("Fault", ['Policy Holder', 'Third Party'])
        with col_c:
            PoliceReportFiled = st.selectbox("Police Report Filed", ['No', 'Yes'])
            WitnessPresent = st.selectbox("Witness Present", ['No', 'Yes'])
        
        st.markdown("---")
        
        # Claim Details
        st.subheader("📝 Claim Details")
        col_a, col_b = st.columns(2)
        with col_a:
            MonthClaimed = st.selectbox("Month Claimed", 
                ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '0'])
        with col_b:
            DayOfWeekClaimed = st.selectbox("Day of Week Claimed", 
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', '0'])
        
        st.markdown("---")
        
        # Agent Information
        st.subheader("👔 Agent Information")
        AgentType = st.selectbox("Agent Type", ['External', 'Internal'])
        
        st.markdown("---")
        
        # Numerical Features
        st.subheader("🔢 Numerical Features")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            WeekOfMonth = st.number_input("Week of Month", min_value=1, max_value=5, value=3)
            WeekOfMonthClaimed = st.number_input("Week of Month Claimed", min_value=1, max_value=5, value=3)
        with col_b:
            Age = st.number_input("Age", min_value=0, max_value=80, value=40)
            PolicyNumber = st.number_input("Policy Number", min_value=1, max_value=15420, value=7710)
        with col_c:
            RepNumber = st.number_input("Rep Number", min_value=1, max_value=16, value=8)
            Deductible = st.number_input("Deductible", min_value=300, max_value=700, value=400, step=100)
        with col_d:
            DriverRating = st.number_input("Driver Rating", min_value=1, max_value=4, value=2)
            Year = st.number_input("Year", min_value=1994, max_value=1996, value=1995)
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)
        
        if predict_button:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Month': [Month],
                'WeekOfMonth': [WeekOfMonth],
                'DayOfWeek': [DayOfWeek],
                'Make': [Make],
                'AccidentArea': [AccidentArea],
                'DayOfWeekClaimed': [DayOfWeekClaimed],
                'MonthClaimed': [MonthClaimed],
                'WeekOfMonthClaimed': [WeekOfMonthClaimed],
                'Sex': [Sex],
                'MaritalStatus': [MaritalStatus],
                'Age': [Age],
                'Fault': [Fault],
                'PolicyType': [PolicyType],
                'VehicleCategory': [VehicleCategory],
                'VehiclePrice': [VehiclePrice],
                'Days_Policy_Accident': [Days_Policy_Accident],
                'Days_Policy_Claim': [Days_Policy_Claim],
                'PastNumberOfClaims': [PastNumberOfClaims],
                'AgeOfVehicle': [AgeOfVehicle],
                'AgeOfPolicyHolder': [AgeOfPolicyHolder],
                'PoliceReportFiled': [PoliceReportFiled],
                'WitnessPresent': [WitnessPresent],
                'AgentType': [AgentType],
                'NumberOfSuppliments': [NumberOfSuppliments],
                'AddressChange_Claim': [AddressChange_Claim],
                'NumberOfCars': [NumberOfCars],
                'Year': [Year],
                'BasePolicy': [BasePolicy],
                'PolicyNumber': [PolicyNumber],
                'RepNumber': [RepNumber],
                'Deductible': [Deductible],
                'DriverRating': [DriverRating]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display prediction
            st.markdown("---")
            st.subheader("🎯 Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ **FRAUDULENT CLAIM** (Confidence: {prediction_proba[1]:.2%})")
            else:
                st.success(f"✅ **LEGITIMATE CLAIM** (Confidence: {prediction_proba[0]:.2%})")
            
            # Display probability bar
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Legitimate Probability", f"{prediction_proba[0]:.2%}")
            with col_prob2:
                st.metric("Fraud Probability", f"{prediction_proba[1]:.2%}")
            
            st.markdown("---")
            
            # LIME Explanation
            st.subheader("🔍 LIME Explanation")
            st.write("Understanding which features contributed to this prediction:")
            
            try:
                # Transform input using preprocessing step
                preprocessed_data = model.named_steps['preprocessing'].transform(input_data)
                
                # Ensure preprocessed_data is 2D array for LIME
                if isinstance(preprocessed_data, pd.DataFrame):
                    preprocessed_array = preprocessed_data.values
                else:
                    preprocessed_array = preprocessed_data
                
                # Get prediction function for the model only (after preprocessing)
                def predict_fn(x):
                    return model.named_steps['model'].predict_proba(x)
                
                # Generate LIME explanation
                explanation = lime_explainer.explain_instance(
                    preprocessed_array[0], 
                    predict_fn,
                    num_features=10
                )
                
                # Plot explanation
                fig = explanation.as_pyplot_figure()
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error generating LIME explanation: {e}")
                st.write("Preprocessed data shape:", preprocessed_array.shape if 'preprocessed_array' in locals() else "Not available")

else:
    st.error("⚠️ Model files not found. Please ensure the following files are in the same directory:")
    st.write("- final_model_xgb_tuned_FIX_BGT_20251105_0911.pkl")
    st.write("- line_explainer.dill")