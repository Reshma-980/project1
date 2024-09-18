from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and feature columns
with open(r'C:\Users\acer\Downloads\code\house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

df = pd.read_csv(r'C:\Users\acer\Downloads\code\cleandata.csv')
X = df.drop('price', axis=1)
X = pd.get_dummies(X, columns=['location'], drop_first=True)
feature_columns = X.columns.tolist()

# Extract unique locations for dropdown (without prefix)
location_columns = [col for col in feature_columns if col.startswith('location_')]
locations = [col.replace('location_', '') for col in location_columns]

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = float(request.form['bath'])
    bhk = float(request.form['bhk'])

    # Prepare input for prediction
    x_input = np.zeros(len(feature_columns))
    x_input[0] = sqft
    x_input[1] = bath
    x_input[2] = bhks

    # One-hot encode location
    loc_column = f'location_{location}'
    if loc_column in feature_columns:
        loc_index = feature_columns.index(loc_column)
        x_input[loc_index] = 1

    # Predict the price
    price = model.predict([x_input])[0]

    return render_template('index.html', locations=locations, prediction_text=f'Predicted Price: ${price:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
