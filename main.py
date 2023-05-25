# 1. Library imports
import uvicorn
from fastapi import FastAPI
from variable import carParameter
import numpy as np
import pickle
import pandas as pd

# Creating the app object
app = FastAPI()

with open('car_price.pkl', 'rb') as f:
    predictor = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/{name}')
def get_name(name: str):
    return {'Say something': f'{name}'}

@app.post('/predict')
def predict_carprice(data:carParameter):
    data = data.dict()
       
    sales=data['sales']
    horsepower=data['horsepower']
    width=data['width']
    fuel_efficiency=data['fuel_efficiency']
    manufacturer=data['manufacturer']
    vehicle_type=data['vehicle_type']

    # print(data)
    
    sales_logvalue = np.log(sales)
    print(sales_logvalue)

    # Load the scaler object from file
    

    X = np.array([sales_logvalue, horsepower, width, fuel_efficiency]).reshape(1, -1)
    scaled_data = scaler.transform(X)
    scaled_array = scaled_data.reshape(4)
    # print(scaled_array)

    # Define the first categorical variable
    car_brands = ['Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet',
        'Chrysler', 'Dodge', 'Ford', 'Honda', 'Hyundai', 'Infiniti',
        'Jaguar', 'Jeep', 'Lexus', 'Lincoln', 'Mitsubishi', 'Mercury',
        'Mercedes-B', 'Nissan', 'Oldsmobile', 'Plymouth', 'Pontiac',
        'Porsche', 'Saab', 'Saturn', 'Subaru', 'Toyota', 'Volkswagen',
        'Volvo']
    
    car_brand_series = pd.Series([manufacturer])   #pass the manufacturer type here
    car_brand_categorical = pd.Categorical(car_brand_series, categories=car_brands)
    car_brand_df = pd.get_dummies(car_brand_categorical)

    # Define the second categorical variable
    vehicle_types = ['Car', 'Passenger']
    vehicle_type_series = pd.Series([vehicle_type])  #pass the vehicle type here
    vehicle_type_categorical = pd.Categorical(vehicle_type_series, categories=vehicle_types)
    vehicle_type_df = pd.get_dummies(vehicle_type_categorical)

    merged_df = pd.concat([car_brand_df, vehicle_type_df], axis=1)
    categorical_array = merged_df.to_numpy()[0]
    # print(categorical_array.shape)
    combined_array = np.concatenate([scaled_array, categorical_array])
    print(combined_array)


    print(combined_array.shape)
    
    prediction_value = predictor.predict([combined_array])
    a = str(prediction_value)
    return {
        'Car PricePrediction': a
    }


    # return prediction

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
