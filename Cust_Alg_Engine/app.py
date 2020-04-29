#imports for solving the ML problem
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler#Normalizing the values
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
#imports for the algorithms
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
#from timer import Timer #self built module to report time
import time as t

#t=Timer()
scaler = StandardScaler()
svc=LinearSVC()
logmodel = LogisticRegression(solver='liblinear')
knn = KNeighborsClassifier()
dtc = tree.DecisionTreeClassifier()
clf = MLPClassifier(solver='lbfgs', activation='logistic', max_iter=100)
Algorithm=[knn, logmodel, svc, dtc, clf]
Algorithm_used=['K Nearest Neighbor', 'Logistic Regression', 'Linear SVM', 'Decision Tree', 'MLP Classifier']

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
            
            #Get the encoding of the file
            with open(os.path.join(app.config["UPLOADS"], 'data.csv')) as f:
                encoding = (str(f).split('encoding=\'')[1]).split('\'>')[0]
            
            #defining the variables as global
            global data
            global column_names
            global outcome
            
            data = pd.read_csv(os.path.join(app.config["UPLOADS"], 'data.csv'), encoding=encoding)
            column_names = data.columns[:-1]
            outcome = data.columns[-1]
            #render the preferences page to choose the preferences based on the data from dataset
            return render_template("preferences.html", columns=column_names,outcome=outcome, data=data.head().values)
        
        #when file extension is not supported
        flash('File not with correct extension')
        return redirect(url_for("index"))
        

@app.route('/preferences', methods=['GET', 'POST'])
def set_preferences():
    if request.method == 'POST':
        
        #Reading data from html form from preferences.html
        algorithm = int(request.form["Algorithm"])
        testsize = (int(request.form["testsize"])/10)
        columns = [int(x) for x in request.form.getlist("selected_columns")]#while using checkboxes in preferences.html
        outcome = request.form['outcome']
        
        columns = [column_names[x-1] for x in columns]
        X = data[columns]
        y = data[outcome]
        labels = y.unique()
        
        #testsize to be given by user read from form data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=101)
        
        #training the model, predicting values, and get the time in doing so
        s = t.perf_counter()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        Algorithm[algorithm].fit(X_train,y_train)
        pred = Algorithm[algorithm].predict(X_test) 
        time_taken = t.perf_counter() - s
        
        #show the rsults page with the metrics
        return render_template('results.html', Algorithm_used=str(Algorithm_used[algorithm]), selected_columns=columns, time_taken=time_taken, confusion_matrix=confusion_matrix(y_test,pred), labels = labels, cr=classification_report(y_test, pred, output_dict=True), accuracy=accuracy_score(y_test, pred), X_test=X_test[:5], y_test=y_test.head().values, pred=pred[0:5])
    
    #when GET request is made
    return "Wrong Method"


app.run()