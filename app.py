from flask import Flask, render_template, request,flash
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib

app = Flask(__name__)


CSV_FILE = 'titanic.csv'

@app.route('/')
def predict():
    return render_template('predict.html')


@app.route('/result', methods=['POST'])
def result():
    age = request.form['Age']
    gender = request.form['Sex']
    pclass = request.form['Pclass']
    fare = request.form['Fare']
    sibsp = request.form['sibsp']
    model=joblib.load('titanic_predict.joblib')
    predictions= model.predict([[age, fare, gender, sibsp, 1, pclass, 2]])
    # print(predictions)

    if predictions == 1:
       return render_template('result.html',  data="Survived")
    else:
       return render_template('result.html',  data="Not survived")


@app.route('/csv')
def display_csv():
    csv_data = pd.read_csv(CSV_FILE)

    selected_columns = ['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked','Survived']
    selected_headers = [header for header in csv_data.columns if header in selected_columns]
    data = csv_data[selected_headers].values.tolist()
    
    X=csv_data.drop(columns=['Survived'])
    Y=csv_data['Survived'] 

    X = csv_data[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']]
    y = csv_data['Survived']

    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X,Y) 
    joblib.dump(model, 'titanic_predict.joblib')

    return render_template('display.html', headers=selected_headers, data=data)


if __name__ == '__main__':
    app.run(debug=True)
