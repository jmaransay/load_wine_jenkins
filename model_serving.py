from typing import List

import contextlib
import requests
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


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass
        
    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

config = uvicorn.Config("example:app", host="127.0.0.1", port=9000, log_level="info")
server = uvicorn.Server(config=config)

with server.run_in_thread():
    # Server started.
    address, port = sock.getsockname()
    print(f"HTTP server is now running on http://{address}:{port}")
    # The API endpoint
    url = "http://127.0.0.1:9000/predict"
    # Data to be sent
    data = {"features": [13.2, 2.77, 2.51, 18.5, 103.0, 1.15, 2.61, 0.26, 1.46, 3.0, 1.05, 3.33, 820.0]}
    # A POST request to the API
    response = requests.post(url, json=data)
    # Print the response
    print(response.json())
    # Data to be sent
    data = {"features": [10.2, 2.1, 3, 14.5, 90.0, 1.34, 2.90, 0.12, 1.99, 1.0, 1.65, 3.88, 700.0]}
    # A POST request to the API
    response = requests.post(url, json=data)
    # Print the response
    print(response.json())
    # Server stopped.




#config = uvicorn.Config(app=app)
#server = uvicorn.Server(config=config)
#(sock := socket.socket()).bind(("127.0.0.1", 9000))
#thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
#thread.start()  # non-blocking call

#while not server.started:
#    time.sleep(0.001)

#address, port = sock.getsockname()
#print(f"HTTP server is now running on http://{address}:{port}")

# The API endpoint
#url = "http://127.0.0.1:9000/predict"

# Data to be sent
#data = {"features": [13.2, 2.77, 2.51, 18.5, 103.0, 1.15, 2.61, 0.26, 1.46, 3.0, 1.05, 3.33, 820.0]}

# A POST request to the API
#response = requests.post(url, json=data)

# Print the response
#print(response.json())

# Data to be sent
#data = {"features": [10.2, 2.1, 3, 14.5, 90.0, 1.34, 2.90, 0.12, 1.99, 1.0, 1.65, 3.88, 700.0]}

# A POST request to the API
#response = requests.post(url, json=data)

# Print the response
print(response.json())
# if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=9000)
