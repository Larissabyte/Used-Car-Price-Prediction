import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Define the exchange rate (1 INR = 0.043 AED as of the last update, please check for the current rate)
exchange_rate_inr_to_aed = 0.043

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocess data
df['Mileage'] = df['Mileage'].str.extract(r'([0-9.]+)').astype(float)  # Extract numeric part
df['Engine'] = df['Engine'].str.extract(r'([0-9.]+)').astype(float)
df['Power'] = df['Power'].str.extract(r'([0-9.]+)').astype(float)

# Drop rows with missing or invalid values
df.dropna(inplace=True)

# Check if Price column needs scaling (if prices are in thousands or lakhs)
if df['Price'].mean() < 100:  # Assuming prices are in lakhs
    df['Price'] *= 100000  # Convert to actual values in rupees

# Select features and target
X = df[['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Fuel_Type', 'Transmission']]
y = df['Price']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['Fuel_Type', 'Transmission'], drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit App
st.title('Used Car Price Prediction App')

# User Input Form
with st.form(key='car_details_form'):
    year = st.slider('Year of Manufacture', 1990, 2024, 2015)
    kilometers_driven = st.number_input('Kilometers Driven', min_value=0, value=50000)
    mileage = st.number_input('Mileage (in kmpl or km/kg)', min_value=0.0, value=18.0, step=0.1)
    engine = st.number_input('Engine Size (in CC)', min_value=500, value=1200, step=100)
    power = st.number_input('Power (in bhp)', min_value=20.0, value=80.0, step=5.0)
    fuel_type = st.selectbox('Fuel Type', options=['Petrol', 'Diesel', 'CNG', 'Electric'])
    transmission = st.selectbox('Transmission Type', options=['Manual', 'Automatic'])

    # Submit button
    submit_button = st.form_submit_button(label='Predict Price')

if submit_button:
    # Prepare input data
    input_data = pd.DataFrame([{
        'Year': year,
        'Kilometers_Driven': kilometers_driven,
        'Mileage': mileage,
        'Engine': engine,
        'Power': power,
        'Fuel_Type_Diesel': int(fuel_type == 'Diesel'),
        'Fuel_Type_Electric': int(fuel_type == 'Electric'),
        'Fuel_Type_Petrol': int(fuel_type == 'Petrol'),
        'Transmission_Manual': int(transmission == 'Manual')
    }])

    # Reindex input_data to match training features
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Scale input features
    input_data = sc.transform(input_data)

    # Predict price
    predicted_price = model.predict(input_data)[0]

    # Convert prediction to AED
    predicted_price_aed = predicted_price * exchange_rate_inr_to_aed

    # Display the result in AED
    st.success(f'The predicted price of the car is: AED {predicted_price_aed:,.2f}')

st.caption(f'Disclaimer: The predictions are based on a statistical model and are for reference purposes only. Exchange rate used: 1 INR = {exchange_rate_inr_to_aed} AED')
