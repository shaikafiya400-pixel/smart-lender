import numpy as np
import pickle
import pandas
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open('rdf.pkl', 'rb'))
scale = pickle.load(open('scale1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=["POST", "GET"])
def predict():
    return render_template("predict.html")

@app.route('/submit', methods=["POST", "GET"])
def submit():
    input_feature = []


    for x in request.form.values():
        try:
            input_feature.append(int(x))
        except:
            input_feature.append(0)

    input_feature = [np.array(input_feature)]

    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
             'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    data = pandas.DataFrame(input_feature, columns=names)

    data = scale.transform(data)

    prediction = model.predict(data)
    prediction = int(prediction[0])

    if prediction == 0:
        return render_template("submit.html", result="Loan will Not be Approved")
    else:
        return render_template("submit.html", result="Loan will be Approved")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)