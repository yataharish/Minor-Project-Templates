import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

# Assuming 'trans' is the transformer used during model training
# Example: 
# from sklearn.preprocessing import StandardScaler
# trans = StandardScaler()
# trans = joblib.load("transformer.pkl")  # Load your transformer

@app.route('/')
def predict():
    return render_template('manual_predict.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    # Get data from the form and convert to float
    x_test = np.array([[float(x) for x in request.form.values()]])
    
    print('Actual:', x_test)
    
    # Transform the data using the loaded transformer
    x_test = trans.transform(x_test)
    
    print('Transformed:', x_test)
    
    # Predict using the loaded model
    pred = model.predict(x_test)
    
    return render_template('manual_predict.html', prediction_text=('Permanent Magnet surface temperature:', pred[0]))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
