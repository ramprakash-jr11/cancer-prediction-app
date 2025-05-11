from flask import Flask, render_template, request, redirect, url_for, session, make_response
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key for production

# Load the trained model
model = joblib.load('rf_cancer_model.pkl')

# Disable caching for development
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        # Get form data
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        smoking = int(request.form['smoking'])
        genetic_risk = int(request.form['genetic_risk'])
        physical_activity = float(request.form['physical_activity'])
        alcohol_intake = float(request.form['alcohol_intake'])
        cancer_history = int(request.form['cancer_history'])

        # Create input array for prediction
        input_data = np.array([[age, gender, bmi, smoking, genetic_risk, 
                                physical_activity, alcohol_intake, cancer_history]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_text = "Cancer (1)" if prediction == 1 else "No Cancer (0)"

        return render_template('result.html', prediction=prediction_text)
    except KeyError as e:
        return render_template('result.html', prediction=f"Error: Missing field {str(e)}")
    except ValueError as e:
        return render_template('result.html', prediction=f"Error: Invalid input value")
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

@app.route('/result')
def result():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('result.html', prediction=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password123':  # Replace with secure auth in production
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid username or password!')
    return render_template('login.html', error=None)

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)