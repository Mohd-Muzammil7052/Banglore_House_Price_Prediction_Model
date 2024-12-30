from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

pipe = pickle.load(open('RidgeModel.pkl','rb'))
data = pd.read_csv('Cleaned_data.csv')



@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations )

@app.route('/predict',methods=['POST'])
def predict():
    # Extract inputs from the form (ensure the order matches the pipeline)
    # Extract inputs from the form
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = float(request.form['bath'])
    sqft = float(request.form['total_sqft'])

    # Prepare input
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    print(f"Input: {input}")  # Debugging

    # Use the pipeline for prediction
    try:
        prediction = pipe.predict(input)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "An error occurred during prediction!", 500

    return str(np.round(prediction, 2)*100000)
if __name__=="__main__":
    app.run(debug=True,port=5001)