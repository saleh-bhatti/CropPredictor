# app.py

from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


app = Flask(__name__)

# Load the dataset
data = pd.read_csv('Crop_Recommendation.csv')

# Encode the target variable
label_encoder = LabelEncoder()
data['Crop'] = label_encoder.fit_transform(data['Crop'])

# Define features and target variable
features = ['Nitrogen', 'Phosphorus', 'Potassium',
            'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

# Train the model
X = data[features]
y = data['Crop']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict crop


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    input_data = scaler.transform(
        [[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    prediction = model.predict(input_data)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    return crop


# Function to make TOP 3 predictions for new data
def predict_top_crops(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    new_data = pd.DataFrame([[nitrogen, phosphorus, potassium,
                            temperature, humidity, pH, rainfall]], columns=features)
    new_data = scaler.transform(new_data)
    probabilities = model.predict_proba(new_data)[0]
    # Get indices of top 3 probabilities
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_crops = label_encoder.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]
    return list(top_crops)
    # return list(zip(top_crops, top_probs))


# Plot features
def plot_feature_importances():
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])

    # Save the plot as a file
    plot_path = os.path.join('static', 'feature_importances.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


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
            recommended_crop = predict_top_crops(
                nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall)

            # Generate and save the feature importances plot
            plot_path = plot_feature_importances()

            return render_template('results.html', crop=recommended_crop, plot_url=plot_path)

        except Exception as e:
            error_message = f"Error: {e}"
            return render_template('error.html', message=error_message)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
