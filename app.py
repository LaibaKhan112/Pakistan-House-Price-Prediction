import streamlit as st
import joblib
import pandas as pd

# Load the trained model and data
df = pd.read_csv("Cleaned_House.csv")
df["Area sqft"] = df["Area Size"] * 225
model = joblib.load("house_price_predictor.sav")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .form-title {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 20px;
    }
    .form-container {
        max-width: 600px;
        margin: 0 auto;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the input form
def main():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="form-title">House Price Predictor</div>', unsafe_allow_html=True)

    # Select City
    city = st.selectbox('Select City',df["city"].unique())

    # Filter locations based on selected city
    if city:
        locations = df[df['city'] == city]['location'].unique()
        location = st.selectbox("Select Location", locations)

    # Filter purposes based on selected location
    if location:
        purposes = df[(df['city'] == city) & (df['location'] == location)]['purpose'].unique()
        purpose = st.selectbox("Select Purpose", purposes)

    # Filter property types based on selected location and purpose
    if purpose:
        property_types = df[(df['city'] == city) & (df['location'] == location) & (df['purpose'] == purpose)]['property_type'].unique()
        property_type = st.selectbox("Select Property Type", property_types)

    # Other Inputs
    #area_type = st.selectbox("Area Type", df["Area Type"].unique(), key='area_type')
    Area_sqft = st.number_input("Area Size (sqft)", min_value=0, step=1, key='area_size')
    #latitude = st.number_input("Latitude", min_value=0, step=1, key='latitude')
    #longitude = st.number_input("longitude", min_value=0, step=1, key='longitude')
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1, key='bedrooms')
    baths = st.number_input("Number of Bathrooms", min_value=0, step=1, key='bathrooms')

    # Form submit button
    submit_button = st.button(label='Predict')

    # Make prediction if form is submitted
    if submit_button:
        try:
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'property_type': [property_type],
                'location': [location],
                'city': [city],
                'purpose': [purpose],
                'Area sqft': [Area_sqft],
                'bedrooms': [bedrooms],
                'baths': [baths]
            })
            
            
            # Apply the same preprocessing as done during model training
            
            #input_data_transformed = column_trans.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data)

            # Display the prediction
            st.success(f"Predicted House Price: {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
