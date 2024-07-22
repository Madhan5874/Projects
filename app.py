from flask import Flask,render_template,request
import pickle
app=Flask(__name__)
model=pickle.load(open('diabetes_predict.pkl','rb'))

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def diabetes_predict():
    pregnencies=float(request.form['pregnencies'])
    glucose=float(request.form['glucose'])
    bloodpressure=float(request.form['bloodpressure'])
    skinthickness=float(request.form['SkinThickness'])
    insulin=float(request.form['insulin'])
    bmi=float(request.form['BMI'])
    diabetespedigreefunction=float(request.form['DiabetesPedigreeFunction'])
    age=int(request.form['age'])

    predict=model.predict([[pregnencies,glucose,bloodpressure,skinthickness,insulin
                            ,bmi,diabetespedigreefunction,age]])
    
    return render_template('index.html',diabetes_predict=predict)


if __name__ == '__main__':
    app.run(debug=True)