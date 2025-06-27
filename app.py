'''from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_holiday = pickle.load(open('le_holiday.pkl', 'rb'))
le_weather = pickle.load(open('le_weather.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received contact message from {name} ({email}): {message}")
        return render_template('contact.html', message_sent=True)
    return render_template('contact.html')


@app.route('/inspect')
def inspect():
    return render_template('inspect.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Safely get inputs with defaults
        holiday_raw = request.form.get('Holiday',None)
        weather_raw = request.form.get('weather','Clouds')

        # Validate against encoder
       
        # Encode
        holiday = le_holiday.transform([holiday_raw])[0]
        weather = le_weather.transform([weather_raw])[0]

        # Get and convert remaining fields
        temp = float(request.form['Temp'])
        rain = float(request.form['Rain'])
        snow = float(request.form['snow'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Combine & scale
        features = [holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds]
        scaled_features = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled_features)
        result = f'Predicted Traffic Volume: {prediction[0]:.2f}'

    except Exception as e:
        result = f'Error: {str(e)}'

    return render_template('inspect.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
    '''
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le_holiday = pickle.load(open('le_holiday.pkl', 'rb'))
le_weather = pickle.load(open('le_weather.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received contact message from {name} ({email}): {message}")
        return render_template('contact.html', message_sent=True)
    return render_template('contact.html')

@app.route('/inspect')
def inspect():
    return render_template('inspect.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve raw inputs
        holiday_raw = request.form.get('Holiday')
        weather_raw = request.form.get('weather')

        # Ensure fallback values to avoid LabelEncoder errors
        if holiday_raw not in le_holiday.classes_:
            holiday_raw = 'None'  # Default holiday class
        if weather_raw not in le_weather.classes_:
            weather_raw = 'Clouds'  # Default weather class

        # Encode
        holiday = le_holiday.transform([holiday_raw])[0]
        weather = le_weather.transform([weather_raw])[0]

        # Convert numeric inputs
        temp = float(request.form['Temp'])
        rain = float(request.form['Rain'])
        snow = float(request.form['snow'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Final feature vector
        features = [holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds]
        scaled_features = scaler.transform([features])

        # Make prediction
        prediction = model.predict(scaled_features)
        result = f'Predicted Traffic Volume: {prediction[0]:,.2f}'

    except Exception as e:
        result = f'Error: {str(e)}'

    # Render separate result page
    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)

