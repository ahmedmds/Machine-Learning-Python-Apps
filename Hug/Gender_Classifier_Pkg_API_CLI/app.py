import hug
import joblib

import warnings
warnings.filterwarnings("ignore")

gender_vectorizer = open("models/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model = open("models/gender_nv_model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)

@hug.cli()
@hug.get('/predict', examples="name=Mark")
@hug.local()
def predict(name:hug.types.text):
    """Get gender prediction by first name"""
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)
    if prediction[0] == 0:
        gender = "female"
    else:
        gender = "male"
    return {"original_name": gender}

# Expose as local package
# To run, open Terminal and then Python interpreter:
# >>> from app import get_books
# >>> get_books("Book Name")
# {'title': 'BOOK NAME'}
# >>> exit()

# Expose as API
# To run, open Terminal and:
# (venv) ...\Hug_intro>hug -f app.py

# Expose as CLI
# To run, open Terminal and activate environment:
# (venv) ...\Hug_intro>hug -f app.py -c help
# (venv) ...\Hug_intro>hug -f app.py -c get_books "Book Name"

if __name__ == '__main__':
    predict.interface.cli()