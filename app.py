from unittest import TestResult, result
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


app = Flask(__name__)

predictResult=pickle.load(open("spam.pkl","rb"))
vectorizer = CountVectorizer()
vectorizer.fit(predictResult[0])
xt=vectorizer.transform(predictResult[0])
predictResult.pop(0)
yt=predictResult[0]
predictResult.pop(0)
class1 =GradientBoostingClassifier()
class2 = XGBClassifier()
class3 = CatBoostClassifier()
class1.fit(xt,yt)
class2.fit(xt,yt)
class3.fit(xt,yt)

@app.route('/')
def man():
    return render_template("index.html")

@app.route("/predict", methods=["POST","GET"])
def home():
    msg=request.form.get("message")
    msg=[msg]
    xv=vectorizer.transform(msg)
    TestResult=[]
    c1=0
    c2=0
    classifierList=["Neural Network","Logistic Regression","ExtraTrees Classifier","Decision Tree","RandomForest","Naive Bayes","K - Neighbours","Support Vector Machine","Ada Boost Classifier","Gradient Boosting Classifier","XGB","CatBoost"]
    j=0
    for i in predictResult:
        if classifierList[j]=="Gradient Boosting Classifier":
            predictions=class1.predict(xv)
            TestResult.append((classifierList[j],predictions[0])) 
            j=j+1
            predictions=class2.predict(xv)
            
            TestResult.append((classifierList[j],predictions[0]))
            j=j+1
            predictions=class3.predict(xv)
            TestResult.append((classifierList[j],predictions[0]))  
            break
        predictions = i.predict(xv)
        TestResult.append((classifierList[j],predictions[0]))       
        j=j+1
    return render_template("index.html",data=TestResult)

if __name__ =="__main__":
    app.debug=True
    app.run()
