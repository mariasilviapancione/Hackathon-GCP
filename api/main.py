from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd

app = FastAPI()

# Carica modello e label encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Definizione input
class MushroomInput(BaseModel):
    odor: str = Field(..., example="p")
    bruises: str = Field(..., example="t")
    cap_shape: str = Field(..., example="x")
    gill_color: str = Field(..., example="k")
    ring_type: str = Field(..., example="p")

@app.post("/predict")
def predict(input_data: MushroomInput):
    try:
        # 1. Converte i dati input in DataFrame
        input_dict = input_data.dict()
        df_input = pd.DataFrame([input_dict])

        # 2. Applica i label encoders solo sulle 5 feature
        for col in df_input.columns:
            df_input[col] = label_encoders[col].transform(df_input[col])

        # 3. Predice la classe
        pred_encoded = model.predict(df_input)[0]
        pred_label = label_encoders["class"].inverse_transform([pred_encoded])[0]

        return {"prediction": pred_label}
    except Exception as e:
        print("❌ Errore interno:", e)
        return {"error": str(e)}


