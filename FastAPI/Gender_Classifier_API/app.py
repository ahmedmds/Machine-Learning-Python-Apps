import uvicorn
from fastapi import FastAPI, Query
import joblib, os

# Vectorizer
gender_vectorizer = open("models/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

# Models
gender_nv_model = open("models/gender_nv_model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)

app = FastAPI()

@app.get('/')
async def index():
    return {"text": "Hello API Builder!"}

@app.get('/items/')
async def get_items(name:str = Query(None, min_length=2, max_length=7)):
    return {"name": name}

# Machine learning aspect
@app.get('/predict/')
async def predict(name:str = Query(None, min_length=2, max_length=7)):
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)
    if prediction[0] == 0:
        result = "female"
    else:
        result = "male"
    return {"orig_name": name, "prediction": result}

# Alternative to GET, but same results
@app.post('/predict/{name}')
async def predict(name):
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)
    if prediction[0] == 0:
        result = "female"
    else:
        result = "male"
    return {"orig_name": name, "prediction": result}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

