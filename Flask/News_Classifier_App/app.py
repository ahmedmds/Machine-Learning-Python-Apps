from flask import Flask, render_template, url_for, request, flash, redirect
import os
import joblib

news_vectorizer_file = open(os.path.join("static/models/final_news_cv_vectorizer.pkl"), "rb")
news_cv = joblib.load(news_vectorizer_file)

app = Flask(__name__)

def get_keys(pred_val, labels_dict):
    for key, value in labels_dict.items():
        if pred_val == value:
            return key

# Random secret key generated for session data
app.config['SECRET_KEY'] = os.urandom(24)

# Bootstrap example https://getbootstrap.com/docs/4.5/examples/
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':        
        rawtext = request.form['rawtext']
        if not rawtext:
            flash('Please enter text', 'danger')
            return redirect('/')

        vectorized_text = news_cv.transform([rawtext]).toarray()
        
        try:
            modelchoice = request.form['modelchoice']
            if modelchoice == 'nb':
                model_file = open(os.path.join("static/models/newsclassifier_NB_model.pkl"), "rb")
            elif modelchoice == 'logit':
                model_file = open(os.path.join("static/models/newsclassifier_Logit_model.pkl"), "rb")
            elif modelchoice == 'rf':
                model_file = open(os.path.join("static/models/newsclassifier_RFOREST_model.pkl"), "rb")
            model = joblib.load(model_file)
            prediction_labels_dict = {'Business': 0, 'Tech': 1, 'Sports': 2, 'Health': 3, 'Politics': 4, \
                                        'Entertainment': 5}
            prediction_val = model.predict(vectorized_text)
            prediction = get_keys(prediction_val, prediction_labels_dict)
            return render_template('index.html', prediction=prediction, rawtext=rawtext)
        except:
            flash('Please select model', 'danger')
            return redirect('/')
    

if __name__ == '__main__':
    app.run(debug=True)