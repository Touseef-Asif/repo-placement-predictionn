from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form['cgpa'])
    resume_score = float(request.form['resume_score'])
    
    features = np.array([[cgpa, resume_score]])
    
    prediction = model.predict(features)[0]
    output = 'Placed' if prediction == 1 else 'Not Placed'
    
    return render_template('index.html', prediction_text=f'The student is {output}')

if __name__ == '__main__':
    app.run(debug=True)
