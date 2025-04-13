import requests
import pandas as pd
import pickle

# Carica dati grezzi e label encoders
df = pd.read_csv("../data/mushrooms.csv")
sample = df.iloc[0].copy()

# Prepara input per FastAPI
input_dict = {}
for col in sample.index:
    if col != "class":
        new_key = col.replace("-", "_")
        input_dict[new_key] = str(sample[col])  # ATTENZIONE: valore come stringa

# Debug del payload inviato
print("Payload JSON:", input_dict)

# Invia richiesta all'API
url = 'http://127.0.0.1:8000/predict/'
response = requests.post(url, json=input_dict)

print("Status code:", response.status_code)
print("Response text:", response.text)

try:
    print("Prediction:", response.json())
except Exception as e:
    print("❌ ERRORE nel parsing JSON:", e)
