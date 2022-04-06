from fastapi import FastAPI
import uvicorn

# Initialize app
app = FastAPI()

# Route
@app.get('/')
async def index():
    return {"text": "Hello API Builder!"}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
