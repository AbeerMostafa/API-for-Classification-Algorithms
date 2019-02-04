from flask import Flask, render_template, request
import sklearn.linear_model  as lm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier



app = Flask(__name__)

@app.route('/')
@app.route('/user',methods = ['POST', 'GET'])
def user():
    if request.method == 'POST':
        result = request.form
    return render_template('user.html')

@app.route('/acc',methods = ['POST', 'GET'])
def acc():
    if request.method == 'POST':
        if request.form['Algorithm'] == 'KNN':
            result = KNN()
        elif request.form['Algorithm'] == 'SVM':
            result = SVM()
        elif request.form['Algorithm'] == 'Decision Tree':
            result = DecisionTree()
        elif request.form['Algorithm'] == 'Naive Bayes':
            result = NaiveBayes()
        elif request.form['Algorithm'] == 'Random Forest':
            result = RandomForest()
        elif request.form['Algorithm'] == 'Logistic Regression':
            result = LogisticRegression()
        else:
            result = {'Error': 'No Valid Algorithm'}
    return render_template("acc.html",result = result)


@app.route('/user2',methods = ['POST', 'GET'])
def user2():
    if request.method == 'POST':
        result = request.form
    return render_template("user2.html",result = result)


@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        predict = request.form
    
    new_x = [[]]  
    
    new_x[0].append(1 if predict['Gender'] == 'Male' else 0 )
    new_x[0].append(1 if predict['SeniorCitizen'] == 'Yes' else 0 )
    new_x[0].append(1 if predict['Dependents'] == 'Yes' else 0 )
    new_x[0].append(2 if predict['MultipleLines'] == 'Yes' else 0 if predict['MultipleLines'] == 'No' else 1 )
    new_x[0].append((int(predict['tenure'])-33)/24.38)
    new_x[0].append((int(predict['TotalCharges'])-2283.3)/2265)

    res = 0
    if get_object.predict(new_x) == 1:
        res = 'Yes'
    elif get_object.predict(new_x) == 0:
        res = 'No'
    
    result = {'Churn': res}
    return render_template("result.html",result = result)



get_object = None



def data_preprocessing():
    
    # Load the Dataset

    df = pd.read_csv("dataset.csv")


    # Feature Selection

    features = ['customerID', 'gender', 'SeniorCitizen', 'Dependents', 'MultipleLines', 'tenure',
                'MonthlyCharges', 'TotalCharges', 'Churn']

    df = df[features]


    # Missing Values Treatment

    s_mode = df['SeniorCitizen'].mode()[0]
    t_mean = df['tenure'].mean()
    fills  = {"SeniorCitizen":s_mode,"tenure":t_mean}
    df.fillna(fills,inplace=True)


    # Label Encoding

    gen_le = LabelEncoder()
    dep_le = LabelEncoder()
    ml_le  = LabelEncoder()
    churn_le = LabelEncoder()


    df['gender']       = gen_le.fit_transform(df['gender'])
    df['Dependents']   = dep_le.fit_transform(df['Dependents'])
    df['MultipleLines']= ml_le.fit_transform(df['MultipleLines'])
    df['Churn']        = churn_le.fit_transform(df['Churn'])


    # Standardization
    # the equation of standardization: X_stand = X-mean(x)/std(X)
    # apply standardization on the columns: 'tenure' and 'TotalCharges'

    tenure_stand = (df['tenure']-df['tenure'].mean())/df['tenure'].std()
    charges_stand = (df['TotalCharges']-df['TotalCharges'].mean())/df['TotalCharges'].std()
    df['tenure_stand'] = tenure_stand
    df['charges_stand']  = charges_stand



    features_for_classification = ['gender', 'SeniorCitizen', 'Dependents', 'MultipleLines', 'tenure_stand',
                'charges_stand', 'Churn']

    df = df[features_for_classification]

    x = df.iloc[:, [0,1,2,3,4,5]].values
    y = df.iloc[:, 6].values

    # Split Data

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

    # Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, y_train, X_test, y_test

def LogisticRegression():
    X_train, y_train, X_test, y_test = data_preprocessing()
    
    LR_clf = lm.LogisticRegression()
    LR_clf = LR_clf.fit(X_train, y_train)
    y_pred_LR = LR_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_LR)
    score = accuracy_score(y_test, y_pred_LR)
    acc = {'Confusion Matrix': cm, 'Accuracy Score': score}
    
    global get_object
    get_object = LR_clf
    return acc

def KNN():
    X_train, y_train, X_test, y_test = data_preprocessing()
    KNN_clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 3)

    KNN_clf = KNN_clf.fit(X_train, y_train)

    y_pred_KNN = KNN_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_KNN)
    score = accuracy_score(y_test, y_pred_KNN)
    acc = {'Confusion Matrix': cm, 'Accuracy Score': score}
    
    global get_object
    get_object = KNN_clf
    return acc

def SVM():
    X_train, y_train, X_test, y_test = data_preprocessing()
    SVM_clf = SVC(kernel = 'rbf', random_state = 0)

    SVM_clf = SVM_clf.fit(X_train, y_train)

    y_pred_SVM = SVM_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_SVM)
    score = accuracy_score(y_test, y_pred_SVM)
    acc = {'Confusion Matrix': cm, 'Accuracy Score': score}
    
    global get_object
    get_object = SVM_clf
    return acc

def DecisionTree():
    X_train, y_train, X_test, y_test = data_preprocessing()
    DT_clf = tree.DecisionTreeClassifier()
    DT_clf = DT_clf.fit(X_train, y_train)
    y_pred_DT = DT_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_DT)
    score = accuracy_score(y_test, y_pred_DT)
    acc = {'Confusion Matrix': cm, 'Accuracy Score': score}
    
    global get_object
    get_object = DT_clf
    return acc

def NaiveBayes():
    X_train, y_train, X_test, y_test = data_preprocessing()
    NB_clf = GaussianNB()
    NB_clf = NB_clf.fit(X_train, y_train)

    y_pred_NB = NB_clf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_NB)
    score = accuracy_score(y_test, y_pred_NB)
    acc = {'Confusion Matrix': cm, 'Accuracy Score': score}
        
    global get_object
    get_object = NB_clf
    
    return acc

def RandomForest():
    X_train, y_train, X_test, y_test = data_preprocessing()
    RF_clf = RandomForestClassifier(n_estimators=100, max_depth=2)
    RF_clf = RF_clf.fit(X_train, y_train)
    y_pred_RF = RF_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_RF)
    score = accuracy_score(y_test, y_pred_RF)
    acc = {'Confusion Matrix': cm, 'Accuracy Score': score}
    
    global get_object
    get_object = RF_clf
    return acc



if __name__ == '__main__':
    app.run(debug = True)