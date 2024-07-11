import pandas as pd
import pickle as pk
import numpy as np
import streamlit as st

model = pk.load(open('RandomForestModel.pkl','rb'))

st.header('Car Price Prediction ML Model')

data = pd.read_csv('Car Data.csv')

def getBrandName(str):
  str= str.split(' ')[0]
  return (str.strip())
data['name']= data['name'].apply(getBrandName)

name = st.selectbox('Select Car Brand',data['name'].unique())
year = st.number_input("Enter Model Year", min_value=1994, max_value=2024, value=1994,step=None)
km_driven = st.number_input("Enter the KM Driven", min_value=11, max_value=300000, value=11,step=None)
fuel = st.selectbox('Select Fuel type',data['fuel'].unique())
seller_type = st.selectbox('Seller Type',data['seller_type'].unique())
transmission = st.selectbox('Select Transmission Type',data['transmission'].unique())
owner = st.selectbox('Owner Type',data['owner'].unique())
mileage = st.number_input("Enter the Mileage", min_value=5.00, max_value=30.00, value=5.00,step=None)
engine = st.number_input("Engine CC", min_value=700.00, max_value=5000.00, value=700.00,step=None)
max_power = st.number_input("Enter the Max Power", min_value=0.00, max_value=200.00, value=0.00,step=None)
seats = st.number_input("Seat Capacity", min_value=5, max_value=10, value=5,step=None)


if st.button('Predict'):
  input = pd.DataFrame(
    [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])


  input['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],inplace=True)
  input['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace=True)
  input['transmission'].replace(['Manual', 'Automatic'],[0,1],inplace=True)
  input['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[0,1,2],inplace=True)
  input['owner'].replace(['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5],inplace=True)


  car_price = model.predict(input)

  st.markdown('car Price is going to be ' + str(car_price[0]))

   