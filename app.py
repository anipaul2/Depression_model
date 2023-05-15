from flask import Flask, url_for, request, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("depression.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = []
    features = ['Age', 'Feeling sad', 'Irritable towards people', 'Trouble sleeping at night',
                'Problems concentrating or making decision', 'loss of appetite', 'Feeling of guilt']

    errors = []

    for feature in features:
        value = request.form.get(feature)
        if value and value.isdigit():
            int_features.append(int(value))
        else:
            errors.append(f"Invalid input for '{feature}'")

    if errors:
        return render_template('depression.html', errors=errors)
    

    final_features = [np.array(int_features)]
    prediction = model.predict_proba(final_features)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if float(output) > 0.5:
        result_text = f"The patient is depressed.\nPercentage of depression is {float(output) * 100}%"
        result_class = "danger"
    else:
        result_text = f"The patient is not depressed.\nPercentage of depression is {float(output) * 100}%"
        result_class = "safe"

    return render_template('depression.html', result_text=result_text, result_class=result_class)


if __name__ == '__main__':
    app.run(debug=True)