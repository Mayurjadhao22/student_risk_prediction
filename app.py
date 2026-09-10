import os
import sys
import pickle
import numpy as np
from flask import Flask, render_template_string, request

# NumPy 2.x to 1.x Unpickling Compatibility Layer for Vercel Serverless
try:
    import numpy._core.multiarray
except ImportError:
    import numpy.core.multiarray
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Academic Status Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: #0f172a;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.2) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.2) 0px, transparent 50%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
            color: #f8fafc;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 
                0 20px 25px -5px rgba(0, 0, 0, 0.6),
                0 8px 10px -6px rgba(0, 0, 0, 0.4),
                0 0 50px rgba(99, 102, 241, 0.15);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            letter-spacing: 0.025em;
        }

        input, select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.25s ease;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        select option {
            background-color: #0f172a;
            color: #ffffff;
        }

        input:focus, select:focus {
            border-color: #818cf8;
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.2),
                0 0 0 3px rgba(129, 140, 248, 0.25),
                0 4px 12px rgba(129, 140, 248, 0.15);
        }

        .btn-submit {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 
                0 4px 14px rgba(99, 102, 241, 0.4),
                0 0 20px rgba(168, 85, 247, 0.25);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 
                0 6px 20px rgba(99, 102, 241, 0.6),
                0 0 30px rgba(168, 85, 247, 0.45);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
            border: 1px solid rgba(168, 85, 247, 0.35);
            border-radius: 16px;
            text-align: center;
            box-shadow: 
                0 10px 25px -5px rgba(168, 85, 247, 0.25),
                inset 0 1px 1px rgba(255, 255, 255, 0.1);
            animation: fadeIn 0.4s ease-out;
        }

        .result-box h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #c084fc;
            margin-bottom: 0.5rem;
        }

        .result-box .value {
            font-size: 2.25rem;
            font-weight: 800;
            color: #ffffff;
            text-shadow: 0 0 12px rgba(255, 255, 255, 0.3);
        }

        .error-box {
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 12px;
            color: #fca5a5;
            text-align: center;
            font-size: 0.9rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 640px) {
            .grid-form {
                grid-template-columns: 1fr;
            }
            .btn-submit {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Student Performance Assessment</h1>
            <p>Enter academic and background metrics to predict student risk profile</p>
        </div>

        <form method="POST" action="/" class="grid-form">
            <div class="form-group">
                <label for="attendance">Attendance (%)</label>
                <input type="number" step="any" id="attendance" name="attendance" value="{{ inputs.get('attendance', '85') }}" required>
            </div>

            <div class="form-group">
                <label for="study_hours">Weekly Study Hours</label>
                <input type="number" step="any" id="study_hours" name="study_hours" value="{{ inputs.get('study_hours', '12') }}" required>
            </div>

            <div class="form-group">
                <label for="past_failures">Past Failures Count</label>
                <input type="number" step="1" id="past_failures" name="past_failures" value="{{ inputs.get('past_failures', '0') }}" required>
            </div>

            <div class="form-group">
                <label for="assignments_completed_pct">Assignments Completed (%)</label>
                <input type="number" step="any" id="assignments_completed_pct" name="assignments_completed_pct" value="{{ inputs.get('assignments_completed_pct', '90') }}" required>
            </div>

            <!-- Categorical Select Option -->
            <div class="form-group">
                <label for="parental_education">Parental Education Category</label>
                <select id="parental_education" name="parental_education" required>
                    <option value="0" {% if inputs.get('parental_education') == '0' %}selected{% endif %}>High School / Below</option>
                    <option value="1" {% if inputs.get('parental_education') == '1' or not inputs %}selected{% endif %}>College / Associate</option>
                    <option value="2" {% if inputs.get('parental_education') == '2' %}selected{% endif %}>Bachelor Degree</option>
                    <option value="3" {% if inputs.get('parental_education') == '3' %}selected{% endif %}>Post-Graduate</option>
                </select>
            </div>

            <!-- Categorical Select Option -->
            <div class="form-group">
                <label for="family_income">Family Income Level</label>
                <select id="family_income" name="family_income" required>
                    <option value="0" {% if inputs.get('family_income') == '0' %}selected{% endif %}>Low</option>
                    <option value="1" {% if inputs.get('family_income') == '1' or not inputs %}selected{% endif %}>Medium</option>
                    <option value="2" {% if inputs.get('family_income') == '2' %}selected{% endif %}>High</option>
                </select>
            </div>

            <!-- Categorical Select Option -->
            <div class="form-group">
                <label for="extracurricular">Extracurricular Involvement</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="0" {% if inputs.get('extracurricular') == '0' %}selected{% endif %}>No</option>
                    <option value="1" {% if inputs.get('extracurricular') == '1' or not inputs %}selected{% endif %}>Yes</option>
                </select>
            </div>

            <!-- Categorical Select Option -->
            <div class="form-group">
                <label for="internet_access">Internet Access at Home</label>
                <select id="internet_access" name="internet_access" required>
                    <option value="0" {% if inputs.get('internet_access') == '0' %}selected{% endif %}>No</option>
                    <option value="1" {% if inputs.get('internet_access') == '1' or not inputs %}selected{% endif %}>Yes</option>
                </select>
            </div>

            <div class="form-group">
                <label for="previous_grade">Previous Grade Score</label>
                <input type="number" step="any" id="previous_grade" name="previous_grade" value="{{ inputs.get('previous_grade', '78') }}" required>
            </div>

            <div class="form-group">
                <label for="final_score">Final Assessment Score</label>
                <input type="number" step="any" id="final_score" name="final_score" value="{{ inputs.get('final_score', '82') }}" required>
            </div>

            <button type="submit" class="btn-submit">Evaluate Academic Status</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <h3>Predicted Student Status Category</h3>
            <div class="value">{{ prediction }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-box">
            {{ error }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    inputs = {}

    if request.method == 'POST':
        inputs = request.form.to_dict()
        if model is None:
            error = "Model file ('model.pkl' or 'linear.pkl') failed to load properly."
        else:
            try:
                # Ordering maps to feature_names_in_:
                # attendance, study_hours, past_failures, assignments_completed_pct,
                # parental_education, family_income, extracurricular, internet_access, previous_grade, final_score
                features = [
                    float(request.form.get('attendance', 0)),
                    float(request.form.get('study_hours', 0)),
                    float(request.form.get('past_failures', 0)),
                    float(request.form.get('assignments_completed_pct', 0)),
                    float(request.form.get('parental_education', 0)),
                    float(request.form.get('family_income', 0)),
                    float(request.form.get('extracurricular', 0)),
                    float(request.form.get('internet_access', 0)),
                    float(request.form.get('previous_grade', 0)),
                    float(request.form.get('final_score', 0))
                ]
                
                pred_val = model.predict(np.array([features]))[0]
                prediction = str(pred_val)
            except Exception as e:
                error = f"Prediction Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, error=error, inputs=inputs)

app = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
