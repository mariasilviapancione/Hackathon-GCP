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
    cap_shape: str = Field(..., example="x")
    cap_surface: str = Field(..., example="s")
    cap_color: str = Field(..., example="n")
    bruises: str = Field(..., example="t")
    odor: str = Field(..., example="p")
    gill_attachment: str = Field(..., example="f")
    gill_spacing: str = Field(..., example="c")
    gill_size: str = Field(..., example="n")
    gill_color: str = Field(..., example="k")
    stalk_shape: str = Field(..., example="e")
    stalk_root: str = Field(..., example="e")
    stalk_surface_above_ring: str = Field(..., example="s")
    stalk_surface_below_ring: str = Field(..., example="s")
    stalk_color_above_ring: str = Field(..., example="w")
    stalk_color_below_ring: str = Field(..., example="w")
    veil_type: str = Field(..., example="p")
    veil_color: str = Field(..., example="w")
    ring_number: str = Field(..., example="o")
    ring_type: str = Field(..., example="p")
    spore_print_color: str = Field(..., example="k")
    population: str = Field(..., example="s")
    habitat: str = Field(..., example="u")


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

