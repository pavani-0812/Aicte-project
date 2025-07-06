# ✅ Import Libraries
import pandas as pd
import joblib
import streamlit as st

# ✅ Load model & column structure (cached for speed)
@st.cache_resource
def load_model_and_columns():
    model = joblib.load("pollution_model.pkl")
    model_cols = joblib.load("model_columns.pkl")
    return model, model_cols

model, model_cols = load_model_and_columns()

# ✅ Streamlit User Interface
st.title("💧 Water Pollutants Predictor")
st.write("Predict the water pollutants based on **Year** and **Station ID**")

# ✅ Inputs from User
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# ✅ Prediction Trigger
if st.button('Predict'):
    if not station_id.strip():
        st.warning('⚠️ Please enter a valid Station ID.')
    else:
        try:
            # ✅ Prepare input DataFrame
            input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
            input_encoded = pd.get_dummies(input_df, columns=['id'])

            # ✅ Align with training columns
            for col in model_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_cols]

            # ✅ Show progress during prediction
            with st.spinner("🔄 Predicting pollutant levels..."):
                predicted_pollutants = model.predict(input_encoded)[0]

            # ✅ Display results
            pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
            st.subheader(f"🔍 Predicted pollutant levels for Station ID '{station_id}' in {year_input}:")
            for p, val in zip(pollutants, predicted_pollutants):
                st.write(f"**{p}**: {val:.2f}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")