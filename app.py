import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic.pkl")
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# HTML & CSS Embedded Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Risk Classifier</title>
    <style>
        :root {
            --primary: #4f46e5;
            --background: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--background);
            color: var(--text);
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 650px;
            margin: 0 auto;
            background: var(--card-bg);
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        h2 { text-align: center; color: var(--primary); margin-bottom: 24px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .form-group { margin-bottom: 16px; }
        .form-group.full { grid-column: span 2; }
        label { display: block; font-weight: 600; font-size: 0.875rem; margin-bottom: 6px; }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 0.95rem;
        }
        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { opacity: 0.9; }
        .result {
            margin-top: 24px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 700;
            background: #e0e7ff;
            color: #3730a3;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Student Risk Assessment</h2>
    <form action="/predict" method="POST">
        <div class="grid">
            <div class="form-group">
                <label>Attendance (%)</label>
                <input type="number" step="any" name="attendance" required>
            </div>
            <div class="form-group">
                <label>Study Hours / Week</label>
                <input type="number" step="any" name="study_hours" required>
            </div>
            <div class="form-group">
                <label>Past Failures</label>
                <input type="number" name="past_failures" min="0" required>
            </div>
            <div class="form-group">
                <label>Assignments Completed (%)</label>
                <input type="number" step="any" name="assignments_completed_pct" required>
            </div>
            <div class="form-group">
                <label>Parental Education</label>
                <select name="parental_education" required>
                    <option value="0">High School</option>
                    <option value="1">Bachelor</option>
                    <option value="2">Master</option>
                    <option value="3">Doctorate</option>
                </select>
            </div>
            <div class="form-group">
                <label>Family Income</label>
                <select name="family_income" required>
                    <option value="0">Low</option>
                    <option value="1">Medium</option>
                    <option value="2">High</option>
                </select>
            </div>
            <div class="form-group">
                <label>Extracurricular Activities</label>
                <select name="extracurricular" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Internet Access</label>
                <select name="internet_access" required>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>
            <div class="form-group">
                <label>Previous Grade</label>
                <input type="number" step="any" name="previous_grade" required>
            </div>
            <div class="form-group">
                <label>Final Score</label>
                <input type="number" step="any" name="final_score" required>
            </div>
        </div>
        <button type="submit">Predict Status</button>
    </form>

    {% if prediction %}
    <div class="result">
        Predicted Category: {{ prediction }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return render_template_string(HTML_TEMPLATE, prediction="Model file missing!")

    try:
        # Extract features in exact order expected by logistic.pkl
        features = [
            float(request.form["attendance"]),
            float(request.form["study_hours"]),
            float(request.form["past_failures"]),
            float(request.form["assignments_completed_pct"]),
            float(request.form["parental_education"]),
            float(request.form["family_income"]),
            float(request.form["extracurricular"]),
            float(request.form["internet_access"]),
            float(request.form["previous_grade"]),
            float(request.form["final_score"]),
        ]

        prediction = model.predict([features])[0]
        return render_template_string(HTML_TEMPLATE, prediction=str(prediction))

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

# Vercel entrypoint
app = app

if __name__ == "__main__":
    app.run(debug=True)
