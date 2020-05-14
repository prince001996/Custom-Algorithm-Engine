#imports for solving the ML problem
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler#Normalizing the values
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score#for classification
from sklearn.metrics import max_error,mean_squared_error, r2_score, mean_absolute_error#for regression

#imports for the algorithms
#Classification Algorithms
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier

#Regression Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
#from timer import Timer #self built module to report time
import time as t

#t=Timer()#not using it
scaler = StandardScaler()#to standardize(normalize) the data

#Configuration of classification algorithms
svc=LinearSVC()
logmodel = LogisticRegression(solver='liblinear')
knn = KNeighborsClassifier()
dtc = tree.DecisionTreeClassifier()
mlpc = MLPClassifier(solver='lbfgs', activation='logistic', max_iter=100)
Classification_Algorithm=[knn, logmodel, svc, dtc, mlpc]
Classification_Algorithm_used=['K Nearest Neighbor', 'Logistic Regression', 'Linear SVM', 'Decision Tree', 'MLP Classifier']

#Configuration of regression algorithms
lin = LinearRegression()
dtr = DecisionTreeRegressor()
svr = SVR()
mlpr = MLPRegressor()
Regression_Algorithm=[lin, dtr, svr, mlpr]
Regression_Algorithm_used=['Linear Regression', 'Decision Tree Regressor', 'SVM Regressor', 'MLP Regressor']

#imports for flask, api, and bowser
from flask_caching import Cache
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify, Response, make_response, after_this_request
from werkzeug.utils import secure_filename
import os

#Flask Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Image detection app'
app.config["DEBUG"] = True
#app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
ALLOWED_EXTENSIONS = {'csv'}
app.config["UPLOADS"] = os.path.join("static", "input")
#testsize = 0
#check file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#stop caching in browser
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

#Home page
@app.route('/')
def index():
    return render_template('index.html')

#api for uploading dataset
@app.route('/upload', methods=['GET', 'POST'])
def upload():
        
    if request.method == 'POST':
        file = request.files['file']
        
        #if file has not been selected to upload
        if file.filename == '':
            flash('Please select a file')
            return redirect(url_for("index"))
        
        #if file has been uploaded and satisfies the supported extension
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config["UPLOADS"], 'data.csv'))
            
            choice = int(request.form["Problem"])
            
            #Get the encoding of the file
            with open(os.path.join(app.config["UPLOADS"], 'data.csv')) as f:
                encoding = (str(f).split('encoding=\'')[1]).split('\'>')[0]
            
            #defining the variables as global
            global data
            global column_names
            global outcome
            
            
            data = pd.read_csv(os.path.join(app.config["UPLOADS"], 'data.csv'), encoding=encoding)
            data.drop(data.columns[data.columns.str.contains('unnamed',case = False)], axis=1, inplace=True)
            column_names = data.columns[:-1]
            outcome = data.columns[-1]
            
            #removing last column from both as last column is Target column
            default_categorical_columns = [each for each in data.select_dtypes(include=['object']).columns if each in column_names]#columns of dtype==object
            default_numerical_columns = [each for each in data.select_dtypes(exclude=['object']).columns if each in column_names]#columns of dtype other than object
            
            #selecting type of algorithm
            algorithm = Classification_Algorithm_used if choice == 0 else Regression_Algorithm_used
           
            #rendering the prefernces page with all the provided dynamically
            return render_template("preferences.html", columns=column_names, default_categorical_columns=default_categorical_columns, default_numerical_columns=default_numerical_columns, outcome=outcome, data=data.head().values, algorithm = algorithm, choice=choice)
        
        #when file extension is not supported
        flash('File not with correct extension')
        return redirect(url_for("index"))
              

@app.route('/solve', methods=['GET', 'POST'])
def solve():
    if request.method == 'POST':
        
        #Reading data from html form from preferences.html
        algorithm = int(request.form["Algorithm"])
        testsize = (int(request.form["testsize"])/10)
        #print(testsize)
        #columns = [int(x) for x in request.form.getlist("selected_columns")]#while using checkboxes in preferences.html
        outcome = request.form['outcome']
        choice = int(request.form["choice"])#represents problem type
        
        selected_categorical_columns = request.form.getlist('selected_categorical_columns')
        selected_numerical_columns = request.form.getlist('selected_numerical_columns')

        s = t.perf_counter()
        
        target = data[outcome]#target column of data
        features = data[selected_categorical_columns+selected_numerical_columns]#data after removing target column and with selected categorical and numerical columns
        
        
        #segregating categorical and numerical data and applying One-Hot Encoding
        categorical_data = features[selected_categorical_columns]
        numerical_data = features[selected_numerical_columns]
        
        X = features
        y = target
        labels = y.unique()
        
        #testsize to be given by user read from form data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=101)
        
        testsize = 0
        
        X_train_ohc = pd.get_dummies(data=X_train, columns=selected_categorical_columns)
        X_train_categorical_data_ohc = X_train_ohc[[x for x in X_train_ohc.columns if x not in selected_numerical_columns]]#one hot encoded categorical data
        selected_categorical_columns_ohc = X_train_categorical_data_ohc.columns
        scaler.fit(X_train[selected_numerical_columns])
        temp1 = scaler.transform(X_train[selected_numerical_columns])
        temp2 = X_train_ohc[selected_categorical_columns_ohc].values
        X_train = np.concatenate((temp1, temp2), axis=1)
        
        original_X_test = X_test#to display data on results page
        X_test_ohc = pd.get_dummies(data=X_test, columns=selected_categorical_columns)
        temp1 = scaler.transform(X_test[selected_numerical_columns])
        temp2 = X_test_ohc[selected_categorical_columns_ohc].values
        X_test = np.concatenate((temp1, temp2), axis=1)
        
        
        
        
        
        if choice == 0:#Classification
            
            #training the model, predicting values, and get the time in doing so
            Classification_Algorithm[algorithm].fit(X_train,y_train)
            pred = Classification_Algorithm[algorithm].predict(X_test) 
            time_taken = t.perf_counter() - s
            
            print(original_X_test)
            
            #show the rsults page with the metrics
            return render_template('Classification_results.html', Classification_Algorithm_used=str(Classification_Algorithm_used[algorithm]), selected_columns=selected_categorical_columns+selected_numerical_columns, time_taken=time_taken, confusion_matrix=confusion_matrix(y_test,pred), labels = labels, cr=classification_report(y_test, pred, output_dict=True), accuracy=accuracy_score(y_test, pred), X_test=original_X_test.head().values, y_test=y_test.head().values, pred=pred[0:5])
            
        else :#Regression

            #training the model, predicting values, and get the time in doing so
            Regression_Algorithm[algorithm].fit(X_train,y_train)
            pred = Regression_Algorithm[algorithm].predict(X_test) 
            time_taken = t.perf_counter() - s
            
            #show the rsults page with the metrics
            return render_template('Regression_results.html', Regression_Algorithm_used=str(Regression_Algorithm_used[algorithm]), selected_columns=selected_categorical_columns+selected_numerical_columns, time_taken=time_taken, mean_squared_error=mean_squared_error(y_test,pred), max_error=max_error(y_test, pred), r2_score=r2_score(y_test, pred), X_test=original_X_test.head().values, y_test=y_test.head().values, pred=pred[0:5])
        
    
    #when GET request is made
    return "Wrong Method"

"""
#set testsize using ajax call
@app.route('/set_testsize', methods=['POST','GET'])
def set_testsize():
    global testsize
    if request.method == 'POST':
        data=request.form['data']
        testsize = (int(data)/10)
        return ''
    if request.method=='GET':
        if testsize == 0:
            
            return test_size

"""


app.run()