from unittest import TestResult, result
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics import accuracy_score


app = Flask(__name__)

predictResult=pickle.load(open("spam.pkl","rb"))
vectorizer = CountVectorizer()
vectorizer.fit(predictResult[0])
predictResult.pop(0)


@app.route('/')
def man():
    return render_template("index.html")

@app.route("/predict", methods=["POST","GET"])
def home():
    print(1)
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
            TestResult.append((classifierList[j],1))
            j=j+1
            TestResult.append((classifierList[j],1))
            j=j+1
            TestResult.append((classifierList[j],1))
            break
        predictions = i.predict(xv)
        TestResult.append((classifierList[j],predictions[0]))       
        j=j+1
    return render_template("index.html",data=TestResult)

if __name__ =="__main__":
    app.debug=True
    app.run()
