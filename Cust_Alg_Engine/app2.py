#imports for solving the problem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from timer import Timer #self built module to report time

#imports for flask, api, and bowser
from flask_caching import Cache
from flask import Flask, flash, request, redirect, url_for, render_template, send_file, jsonify, Response, make_response, after_this_request
from werkzeug.utils import secure_filename
import os

t=Timer()
svc1=LinearSVC()
svc2=SVC(gamma='scale')
logmodel = LogisticRegression(solver='liblinear')
knn = KNeighborsClassifier()
Algorithm=[knn, logmodel, svc1]
Algorithm_used=['K Nearest Neighbor', 'Logistic Regression', 'Linear SVM']

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
        
    if request.method == 'POST':
        file = request.files['file']
        
        if file.filename == '':
            flash('Please select a file')
            return redirect(url_for("index"))
            
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config["UPLOADS"], 'data.csv'))

            global data
            global column_names
            data = pd.read_csv(os.path.join(app.config["UPLOADS"], 'data.csv'))
            column_names = data.columns
            
            return render_template("preferences.html", columns=column_names, data=data.head().as_matrix())
       
        flash('File not with correct extension')
        return redirect(url_for("index"))
        

@app.route('/preferences', methods=['GET', 'POST'])
def set_preferences():
    if request.method == 'POST':
        algorithm = int(request.form["Algorithm"])
        testsize = float(request.form["testsize"])
        columns = [int(x) for x in str(request.form["columns"]).split(',')]#while taking it as numbers from prefernces.html
        #columns = [int(x) for x in request.form.getlist("selected_columns")]#while using checkboxes in preferences2.html
        label = int(request.form['label'])
        
        columns = [column_names[x-1] for x in columns]
        X = data[columns]
        #X = data[(column_names[x-1] for x in columns)]
        y = data[column_names[label-1]]
        labels = y.unique()
        
        #testsize to be given by user
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=101)
        
        t.start()
        Algorithm[algorithm].fit(X_train,y_train)
        pred = Algorithm[algorithm].predict(X_test) 
        time_taken = t.stop()
        
        #return 'Algorithm used : '+ str(Algorithm_used[algorithm])+'Columns Selected : '+str(columns)+'\n'+'Confusion Matrix : '+str(confusion_matrix(y_test,pred))+'\n'+'Accuracy Score : '+str(accuracy_score(y_test, pred))+'\n'+'Classification report : '+str(classification_report(y_test, pred))+'\n'+'time_taken : '+time_taken
        
        return render_template('results.html', Algorithm_used=str(Algorithm_used[algorithm]), selected_columns=columns, time_taken=time_taken, confusion_matrix=confusion_matrix(y_test,pred), labels = labels, cr=classification_report(y_test, pred, output_dict=True), accuracy=accuracy_score(y_test, pred), X_test=X_test.head().as_matrix(), y_test=y_test.head().as_matrix(), pred=pred[0:5])
    
    return "Wrong Method"

app.run()