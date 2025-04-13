from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Carica modello e label encoders
model = pickle.load(open("model.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))

# Definizione input
class MushroomInput(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str

@app.post("/predict")
def predict(input_data: MushroomInput):
    try:
        input_dict = input_data.dict()
        df_input = pd.DataFrame([input_dict])

        for col in df_input.columns:
            df_input[col] = label_encoders[col].transform(df_input[col])

        prediction_encoded = model.predict(df_input)[0]
        prediction_label = label_encoders["class"].inverse_transform([prediction_encoded])[0]

        return {"prediction": prediction_label}
    except Exception as e:
        print("❌ Errore interno:", e)
        return {"error": str(e)}

