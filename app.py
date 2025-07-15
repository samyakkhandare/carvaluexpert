from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and encoders
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

# 🚗 Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict_form.html')

    try:
        data = request.form
        input_data = {
            'Brand': data['brand'].strip(),
            'Model': data['model'].strip(),
            'Year': int(data['year']),
            'Fuel_Type': data['fuel_type'],
            'Transmission': data['transmission'],
            'Mileage': float(data['mileage']),
            'Engine_CC': int(data['engine_cc']),
            'Seating_Capacity': int(data['seating_capacity']),
            'Service_Cost': float(data['service_cost'])
        }

        df = pd.DataFrame([input_data])

        # Feature engineering
        df['Mileage_Category'] = df['Mileage'].apply(lambda x: 'High' if x >= 25 else 'Medium' if x >= 15 else 'Low')
        df['Car_Age'] = 2025 - df['Year']
        df['Cost_per_CC'] = 0  # no price input
        df['Seats_Premium'] = df['Seating_Capacity'].apply(lambda x: 1 if x >= 6 else 0)

        for col in ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Mileage_Category']:
            le = label_encoders[col]
            val = df[col].iloc[0]
            if val not in le.classes_:
                le.classes_ = np.append(le.classes_, val)
            df[col] = le.transform(df[col])

        df = df.drop(columns=['Year'], errors='ignore')
        df = df[model.feature_names_in_]

        prediction = round(model.predict(df)[0])

        log_data = input_data.copy()
        log_data['PredictedPrice'] = prediction
        pd.DataFrame([log_data]).to_csv(
            'prediction_log.csv',
            mode='a',
            header=not os.path.exists('prediction_log.csv'),
            index=False
        )

        market_avg = 1600000
        return render_template("result.html", prediction=prediction, car=input_data, market_avg=market_avg)

    except Exception as e:
        return f"<h3>❌ Something went wrong:</h3><pre>{e}</pre>"

# 🧾 View Prediction Logs (with optional search)
@app.route('/view-predictions', methods=['GET', 'POST'])
def view_predictions():
    log_file = 'prediction_log.csv'
    predictions = []
    query = ""

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)

        # Search functionality
        if request.method == 'POST':
            query = request.form['search'].strip().lower()
            df = df[df.apply(lambda row: row.astype(str).str.lower().str.contains(query).any(), axis=1)]

        predictions = df.to_dict(orient='records')

    return render_template("view_predictions.html", predictions=predictions, query=query)

# 🏠 Admin Dashboard
@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template("admin_dashboard.html")

# 📝 Admin Register
@app.route('/admin/register', methods=['GET', 'POST'])
def register_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print(f"Admin Registered: {username} | {password}")
        return render_template("register_success.html", username=username)
    return render_template('register.html')

# 🔐 Admin Login
@app.route('/admin/login', methods=['GET', 'POST'])
def login_admin():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "admin123":
            return render_template("admin_dashboard.html")
        else:
            error = "❌ Invalid credentials"
    return render_template('login.html', error=error)

# 🚪 Logout (simply redirect to home)
@app.route('/admin/logout')
def admin_logout():
    return redirect(url_for('home'))

# ▶️ Run the App
if __name__ == '__main__':
    app.run(debug=True)
