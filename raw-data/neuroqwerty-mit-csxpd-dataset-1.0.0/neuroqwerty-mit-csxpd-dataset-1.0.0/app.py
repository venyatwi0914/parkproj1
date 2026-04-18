import os, re, numpy as np, pandas as pd, joblib, openai
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
K2_API_KEY = "your_k2_api_key_here"
client = openai.OpenAI(api_key=K2_API_KEY, base_url="https://platform.moonshot.ai/v1")

MODEL_PATH, SCALER_PATH = 'pd_random_forest_model.pkl', 'standard_scaler.pkl'
patient_data_store = {} # Temp store for session biomarkers

# --- MEDICAL VAULT & RESEARCH ---
PATIENT_VAULT = {
    "Venya Tiwari": {
        "history": "Age 18, Rutgers Undergraduate. No tremors reported. NJ resident.",
        "risk_factors": "Low pesticide exposure; urban lifestyle."
    },
    "John Doe": {
        "history": "Age 68, Retired farmer. Reports slight resting tremor in right hand.",
        "risk_factors": "Historical exposure to agricultural pesticides (Paraquat)."
    }
}

RESEARCH_INSIGHTS = """
MIT-CS1PD Study: Hold Time (HT) Mean and CV (Coefficient of Variation) 
are primary biomarkers. A significant difference (p=0.018) exists in HT Mean 
between early PD and control groups. High HT variance indicates motor 
control degradation.
"""

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
    return None, None

model, scaler = load_model()

# --- PREPROCESSING ---
def extract_live_features(events):
    p_noise = re.compile(r'mouse.+|Shift|Alt|Control|Meta|Command|BackSpace', re.IGNORECASE)
    cleaned = [ev for ev in events if not p_noise.match(str(ev.get('key', ''))) and 0 <= ev.get('hold_time', 0) < 5]
    if len(cleaned) < 10: return None
    df = pd.DataFrame(cleaned)
    ht, press = df['hold_time'].values, df['press_time'].values
    ft = np.diff(press)
    ft = ft[(ft > 0) & (ft < 5)]
    return {
        'ht_mean': np.mean(ht), 'ht_std': np.std(ht), 'ht_cv': np.std(ht)/(np.mean(ht)+1e-9),
        'ft_mean': np.mean(ft) if len(ft)>0 else 0, 'ft_std': np.std(ft) if len(ft)>0 else 0,
        'typing_speed': len(ht)/(press[-1]-press[0])*60 if len(press)>1 else 0
    }

# --- ENDPOINTS ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data.get('patient_name', 'anonymous')
    features = extract_live_features(data['events'])
    if not features: return jsonify({"error": "Insufficient data"}), 400
    
    res = {"features": features, "probability": 0.0}
    if model and scaler:
        vec = [features[f] for f in ['ht_mean', 'ht_std', 'ht_cv', 'ft_mean', 'ft_std', 'typing_speed']]
        res["probability"] = float(model.predict_proba(scaler.transform([vec]))[0][1])
    
    patient_data_store[name] = res # Save for doctor search
    return jsonify(res)

@app.route('/doctor/search', methods=['POST'])
def search():
    name = request.json.get('name')
    if name in PATIENT_VAULT and name in patient_data_store:
        return jsonify({"history": PATIENT_VAULT[name], "biometrics": patient_data_store[name]})
    return jsonify({"error": "Patient not found"}), 404

@app.route('/doctor/chat', methods=['POST'])
def chat():
    data = request.json
    name, question = data.get('name'), data.get('question')
    ctx = f"Patient: {name}\nHistory: {PATIENT_VAULT[name]}\nBiometrics: {patient_data_store[name]}\n\nResearch: {RESEARCH_INSIGHTS}"
    
    response = client.chat.completions.create(
        model="kimi-k2-thinking",
        messages=[
            {"role": "system", "content": "You are a clinical assistant. Use the research and data provided to reason step-by-step."},
            {"role": "user", "content": f"{ctx}\n\nQuestion: {question}"}
        ]
    )
    return jsonify({"reply": response.choices[0].message.content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))