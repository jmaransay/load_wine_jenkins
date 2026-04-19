from typing import List

import os
import threading
import time
import socket
import joblib
import uvicorn
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define the labels corresponding to the target classes
LABELS = [
    "Verdante",  # A vibrant and fresh wine, inspired by its balanced acidity and crisp flavors.
    "Rubresco",  # A rich and robust wine, named for its deep, ruby color and bold taste profile.
    "Floralis",  # A fragrant and elegant wine, known for its floral notes and smooth finish.
]


class Features(BaseModel):
    features: List[float]


def load_model(file_path):
    return joblib.load(file_path)


model = load_model("model.pkl")


@app.post("/predict")
def predict(features: Features):
    # Get the numerical prediction
    prediction_index = model.predict([features.features])[0]
    # Map the numerical prediction to the label
    prediction_label = LABELS[prediction_index]
    return {"prediction": prediction_label}


config = uvicorn.Config(app=app)
server = uvicorn.Server(config=config)
(sock := socket.socket()).bind(("127.0.0.1", 9000))
thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
thread.start()  # non-blocking call

while not server.started:
    time.sleep(0.001)

address, port = sock.getsockname()
print(f"HTTP server is now running on http://{address}:{port}")

# subprocess.check_output(['curl', '-X', 'POST', 'http://127.0.0.1:9000/predict', '-H', '"Content-Type: application/json"', '-d', '{"features": [13.2, 2.77, 2.51, 18.5, 103.0, 1.15, 2.61, 0.26, 1.46, 3.0, 1.05, 3.33, 820.0]}'])

subprocess.run(['curl', '-X', 'POST', 'http://127.0.0.1:9000/predict', '-H', '"Content-Type: application/json"', '-d', '{"features": [13.2, 2.77, 2.51, 18.5, 103.0, 1.15, 2.61, 0.26, 1.46, 3.0, 1.05, 3.33, 820.0]}'], capture_output=True)

#result = os.popen(['curl', '-X', 'POST', 'http://127.0.0.1:9000/predict', '-H', '"Content-Type: application/json"', '-d', '{"features": [13.2, 2.77, 2.51, 18.5, 103.0, 1.15, 2.61, 0.26, 1.46, 3.0, 1.05, 3.33, 820.0]}']).read()
#print (result)

# if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=9000)
