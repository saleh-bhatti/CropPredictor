from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re

app = Flask(__name__)

def insert_spaces(text):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', text)

data = pd.read_csv('Crop_Recommendation.csv')

# Apply insert_spaces to the Crop column before encoding
data['Crop'] = data['Crop'].apply(insert_spaces)

label_encoder = LabelEncoder()
data['Crop'] = label_encoder.fit_transform(data['Crop'])

features = ['Nitrogen', 'Phosphorus', 'Potassium',
            'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

X = data[features]
y = data['Crop']

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall):
    input_data = scaler.transform(
        [[nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall]])
    prediction = model.predict(input_data)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    return crop

def predict_top_crops(nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, available_crops):
    new_data = pd.DataFrame([[nitrogen, phosphorus, potassium,
                            temperature, humidity, pH, rainfall]], columns=features)
    new_data = scaler.transform(new_data)
    probabilities = model.predict_proba(new_data)[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    top_crops = label_encoder.inverse_transform(sorted_indices)
    top_probs = probabilities[sorted_indices]
    filtered_crops = [(crop, prob) for crop, prob in zip(top_crops, top_probs) if crop in available_crops]
    return filtered_crops[:3]

def plot_feature_importances(top_crops):
    overall_importances = model.feature_importances_
    crop_importances = {crop: np.zeros_like(overall_importances) for crop, _ in top_crops}

    original_y = y.copy()  # Save original labels

    for crop, _ in top_crops:
        crop_index = label_encoder.transform([crop])[0]
        binary_y = (original_y == crop_index).astype(int)
        temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
        temp_model.fit(X, binary_y)
        crop_importances[crop] = temp_model.feature_importances_

    indices = np.argsort(overall_importances)[::-1]

    plt.figure(figsize=(12, 8))
    bar_width = 0.25
    colors = ['#01176B', '#FF837A', '#11FF8F']

    for i, (crop, _) in enumerate(top_crops):
        plt.bar(np.arange(X.shape[1]) + i * bar_width, crop_importances[crop][indices], bar_width, label=crop, color=colors[i])

    plt.title("Feature Importances for Top Predicted Crops")
    plt.xticks(np.arange(X.shape[1]) + bar_width, [features[j] for j in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join('static', 'feature_importances.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    all_crops = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
    all_crops = [insert_spaces(crop) for crop in all_crops]

    if request.method == 'POST':
        try:
            available_crops = request.form.getlist('available_crops')
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pH = float(request.form['pH'])
            rainfall = float(request.form['rainfall'])

            recommended_crops = predict_top_crops(
                nitrogen, phosphorus, potassium, temperature, humidity, pH, rainfall, available_crops)

            plot_path = plot_feature_importances(recommended_crops)

            return render_template('results.html', crops=recommended_crops, plot_url=plot_path)

        except Exception as e:
            error_message = f"Error: {e}"
            return render_template('error.html', message=error_message)

    return render_template('index.html', crops=all_crops)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
