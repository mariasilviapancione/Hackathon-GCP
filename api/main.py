from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd

# Inizializza FastAPI
app = FastAPI()

# Abilita CORS per richieste dal web (es. GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica modello e label encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Modello dati per la richiesta
class MushroomInput(BaseModel):
    odor: str = Field(..., example="p")
    bruises: str = Field(..., example="t")
    cap_shape: str = Field(..., example="x")
    gill_color: str = Field(..., example="k")
    ring_type: str = Field(..., example="p")

# Endpoint di predizione
@app.post("/predict")
def predict(input_data: MushroomInput):
    try:
        # Converte l'input in DataFrame
        input_dict = input_data.dict()
        df_input = pd.DataFrame([input_dict])

        # Applica label encoding
        for col in df_input.columns:
            df_input[col] = label_encoders[col].transform(df_input[col])

        # Predice e decodifica l'output
        pred_encoded = model.predict(df_input)[0]
        pred_label = label_encoders["class"].inverse_transform([pred_encoded])[0]

        return {"prediction": pred_label}
    except Exception as e:
        print("❌ Errore interno:", e)
        return {"error": str(e)}


