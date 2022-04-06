## Serving Our ML Model as an API
+ Using FastAPI
+ Using Flask Jsonify and Swagger

#### Installation
```bash
pip install fastapi
```
or
```bash
pip install fastapi[all]
```


### ASGI server
+ FastAPI requires an asgi server such as
+Uvicorn or Hypercorn.

```bash
pip install uvicorn
```

### Running The API
uvicorn app:app --reload

or

python app.py


### Navigating URL
+ For Docs
http://127.0.0.1:8000/docs

+ For Documentation with ReDoc
http://127.0.0.1:8000/redoc

