# app.py

from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Crop_Recommendation.csv')

# Encode the target variable
label_encoder = LabelEncoder()
data['Crop'] = label_encoder.fit_transform(data['Crop'])

# Define features and target variable
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

# Train the model
X = data[features]
y = data['Crop']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    input_data = scaler.transform([[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    prediction = model.predict(input_data)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    return crop

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pH = float(request.form['pH'])
            rainfall = float(request.form['rainfall'])

            # Predict crop
            recommended_crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall)

            return render_template('results.html', crop=recommended_crop)

        except Exception as e:
            error_message = f"Error: {e}"
            return render_template('error.html', message=error_message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
