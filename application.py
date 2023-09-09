from flask import Flask,request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd

model=pickle.load(open('/config/workspace/model/log_reg.pkl','rb'))
scaler=pickle.load(open('/config/workspace/model/standardscaler.pkl','rb'))


application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predictdata",methods=['GET','POST'])
def predict_data():
    result=""
    
    if request.method=='POST':
        Pregnancies=int(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
        
        if predict[0]==1:
            result="Diabetic"
            
        
        else:
            result="not Diabetic"
        
        return render_template("predict.html",result=result)


       
    
    
    else:
        return render_template('home.html')
   



if __name__=="__main__":
    app.run(host="0.0.0.0")
