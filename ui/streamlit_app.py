import streamlit as st
import requests

st.set_page_config(page_title="🍄 Mushroom Classifier")

st.title("🍄 Mushroom Edibility Predictor")
st.write("Select the mushroom characteristics to find out whether it's edible or poisonous.")

# Selection options (expand as needed)
odor = st.selectbox("Odor", ["p", "a", "l", "n"], format_func=lambda x: {
    "p": "p (pungent)",
    "a": "a (almond)",
    "l": "l (anise)",
    "n": "n (none)"
}[x])

bruises = st.selectbox("Bruises", ["t", "f"], format_func=lambda x: {
    "t": "t (bruises)",
    "f": "f (no bruises)"
}[x])

cap_shape = st.selectbox("Cap Shape", ["x", "f", "k"], format_func=lambda x: {
    "x": "x (convex)",
    "f": "f (flat)",
    "k": "k (knobbed)"
}[x])

gill_color = st.selectbox("Gill Color", ["k", "n", "b"], format_func=lambda x: {
    "k": "k (black)",
    "n": "n (brown)",
    "b": "b (buff)"
}[x])

ring_type = st.selectbox("Ring Type", ["p", "e"], format_func=lambda x: {
    "p": "p (pendant)",
    "e": "e (evanescent)"
}[x])

if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "odor": odor,
        "bruises": bruises,
        "cap_shape": cap_shape,
        "gill_color": gill_color,
        "ring_type": ring_type
    }

    # API URL
    api_url = "http://127.0.0.1:8000/predict"

    try:
        response = requests.post(api_url, json=input_data)
        result = response.json()

        if "prediction" in result:
            pred = result["prediction"]
            if pred == "e":
                st.success("✅ The mushroom is **edible**!")
            else:
                st.error("☠️ The mushroom is **poisonous**!")
        else:
            st.warning(f"❌ API Error: {result}")
    except Exception as e:
        st.error(f"❌ Request error: {e}")
