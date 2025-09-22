import os
import glob
import json
from datetime import datetime
from typing import List, Dict, Any
import base64
import tempfile
import logging
import subprocess
import platform

import pandas as pd
import joblib
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from flask_cors import CORS

# AI Voice/Image imports
try:
    from groq import Groq
    from gtts import gTTS
    import speech_recognition as sr
    from pydub import AudioSegment
    from io import BytesIO
    AI_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"AI features not available: {e}")
    AI_FEATURES_AVAILABLE = False

# Load environment variables for AI APIs
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration / Paths
# -----------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'DATASET', 'dataset.csv')
DESCRIPTIONS_PATH = os.path.join(os.path.dirname(__file__), 'DATASET', 'symptom_Description.csv')
PRECAUTIONS_PATH = os.path.join(os.path.dirname(__file__), 'DATASET', 'symptom_precaution.csv')

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -----------------------------------------------------------------------------
# Artifact Loading
# -----------------------------------------------------------------------------

def _latest_metadata_file() -> str:
    pattern = os.path.join(MODELS_DIR, 'model_metadata_*.json')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No metadata JSON files found in 'models' directory.")
    # sort by timestamp inside filename
    files.sort(reverse=True)
    return files[0]

def load_artifacts() -> Dict[str, Any]:
    meta_file = _latest_metadata_file()
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    model_path = os.path.join(MODELS_DIR, meta['model_file'])
    le_path = os.path.join(MODELS_DIR, meta['label_encoder_file'])

    model = joblib.load(model_path)

    # Try to load label encoder; if missing create surrogate
    try:
        label_encoder = joblib.load(le_path)
    except Exception:
        class DummyLE:
            classes_ = meta['classes']
            def inverse_transform(self, arr):
                return [self.classes_[i] for i in arr]
        label_encoder = DummyLE()

    # Derive symptom (feature) names
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Fallback: parse dataset
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError("Cannot infer feature names; dataset.csv not found.")
        df = pd.read_csv(DATASET_PATH)
        symptom_cols = [c for c in df.columns if c.lower().startswith('symptom')]
        symptoms = set()
        for col in symptom_cols:
            vals = df[col].dropna().astype(str).str.strip()
            symptoms.update(v for v in vals if v and v.lower() != 'nan')
        feature_names = sorted(symptoms)

    return {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'meta': meta
    }

def load_disease_info() -> Dict[str, Dict[str, Any]]:
    """Load disease descriptions and precautions from CSV files."""
    disease_info = {
        'descriptions': {},
        'precautions': {}
    }
    
    # Load descriptions
    try:
        if os.path.exists(DESCRIPTIONS_PATH):
            desc_df = pd.read_csv(DESCRIPTIONS_PATH)
            for _, row in desc_df.iterrows():
                disease = row['Disease']
                description = row['Description']
                disease_info['descriptions'][disease] = description
    except Exception as e:
        print(f"Warning: Could not load descriptions: {e}")
    
    # Load precautions
    try:
        if os.path.exists(PRECAUTIONS_PATH):
            prec_df = pd.read_csv(PRECAUTIONS_PATH)
            for _, row in prec_df.iterrows():
                disease = row['Disease']
                precautions = []
                for i in range(1, 5):  # Precaution_1 to Precaution_4
                    prec_col = f'Precaution_{i}'
                    if prec_col in row and pd.notna(row[prec_col]) and row[prec_col].strip():
                        precautions.append(row[prec_col].strip())
                
                disease_info['precautions'][disease] = precautions
    except Exception as e:
        print(f"Warning: Could not load precautions: {e}")
    
    return disease_info

artifacts = load_artifacts()
disease_info = load_disease_info()
MODEL = artifacts['model']
LABEL_ENCODER = artifacts['label_encoder']
FEATURE_NAMES: List[str] = artifacts['feature_names']

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def build_feature_vector(selected: List[str]) -> pd.DataFrame:
    """Return a single-row DataFrame matching the model's training columns.

    All features initialized to 0; set 1 for each selected symptom present in FEATURE_NAMES.
    """
    data = {name: 0 for name in FEATURE_NAMES}
    for s in selected:
        if s in data:
            data[s] = 1
    return pd.DataFrame([data])


def predict_disease(symptoms: List[str], top_n: int = 3) -> Dict[str, Any]:
    X_row = build_feature_vector(symptoms)
    pred_idx = MODEL.predict(X_row)[0]
    pred_label = LABEL_ENCODER.inverse_transform([pred_idx])[0]

    result = {
        'predicted_disease': pred_label,
        'input_symptoms': symptoms,
    }

    # Add disease information
    if pred_label in disease_info['descriptions']:
        result['description'] = disease_info['descriptions'][pred_label]
    
    if pred_label in disease_info['precautions']:
        result['precautions'] = disease_info['precautions'][pred_label]

    # Probabilities (if supported)
    if hasattr(MODEL, 'predict_proba'):
        proba = MODEL.predict_proba(X_row)[0]
        # Pair probabilities with class labels
        pairs = list(zip(LABEL_ENCODER.classes_, proba))
        pairs.sort(key=lambda x: x[1], reverse=True)
        result['top_predictions'] = []
        for d, p in pairs[:top_n]:
            disease_pred = {
                'disease': d,
                'probability': round(float(p), 4)
            }
            # Add description and precautions for each top prediction
            if d in disease_info['descriptions']:
                disease_pred['description'] = disease_info['descriptions'][d]
            if d in disease_info['precautions']:
                disease_pred['precautions'] = disease_info['precautions'][d]
            result['top_predictions'].append(disease_pred)
    return result

# -----------------------------------------------------------------------------
# AI Voice/Image Functions
# -----------------------------------------------------------------------------

def encode_image_from_bytes(image_bytes):
    """Convert image bytes to base64 encoding for AI analysis."""
    return base64.b64encode(image_bytes).decode('utf-8')

def analyze_image_with_groq(query, encoded_image, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """Analyze medical image using GROQ AI."""
    if not AI_FEATURES_AVAILABLE:
        return "AI features not available. Please install required packages."
    
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

def transcribe_audio_with_groq(audio_bytes, model="whisper-large-v3"):
    """Transcribe audio to text using GROQ."""
    if not AI_FEATURES_AVAILABLE:
        return "AI features not available. Please install required packages."
    
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    language="en"
                )
            return transcription.text
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def text_to_speech_gtts(text, language="en"):
    """Convert text to speech using gTTS and return audio bytes."""
    if not AI_FEATURES_AVAILABLE:
        return None
    
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to temporary file and read bytes
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            tts.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            return audio_bytes
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

# -----------------------------------------------------------------------------
# HTML Template (inline for simplicity)
# -----------------------------------------------------------------------------
BASE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Disease Prediction</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    h1 { margin-bottom: 0.5rem; }
    form { margin-bottom: 1.5rem; }
    select { width: 100%; height: 300px; }
    .result { padding: 1rem; background:#f4f6f8; border:1px solid #ccc; border-radius:8px; }
    .prob-table { border-collapse: collapse; margin-top:0.5rem; }
    .prob-table th, .prob-table td { border:1px solid #ddd; padding:4px 8px; }
    .footer { margin-top:2rem; font-size:0.8rem; color:#666; }
  </style>
</head>
<body>
  <h1>Disease Prediction</h1>
  <p>Select one or more symptoms (Ctrl / Cmd + click for multi-select) and submit.</p>
  <form method="POST" action="{{ url_for('predict_form') }}">
    <label for="symptoms">Symptoms:</label><br />
    <select id="symptoms" name="symptoms" multiple>
      {% for s in symptoms %}
        <option value="{{ s }}" {% if s in selected %}selected{% endif %}>{{ s }}</option>
      {% endfor %}
    </select>
    <div style="margin-top:1rem;">
      <button type="submit">Predict</button>
      <button type="button" onclick="document.getElementById('symptoms').selectedIndex=-1;">Clear Selection</button>
    </div>
  </form>

  {% if result %}
  <div class="result">
    <h2>Prediction</h2>
    <p><strong>Disease:</strong> {{ result.predicted_disease }}</p>
    {% if result.top_predictions %}
      <h3>Top Probabilities</h3>
      <table class="prob-table">
        <tr><th>Disease</th><th>Probability</th></tr>
        {% for row in result.top_predictions %}
          <tr><td>{{ row.disease }}</td><td>{{ '%.2f'|format(row.probability*100) }}%</td></tr>
        {% endfor %}
      </table>
    {% endif %}
    <p><strong>Selected symptoms:</strong> {{ result.input_symptoms|join(', ') if result.input_symptoms else 'None' }}</p>
  </div>
  {% endif %}

  <div class="footer">Model timestamp: {{ meta.created }} | Features: {{ feature_count }}</div>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('predict_form'))

@app.route('/predict', methods=['GET', 'POST'])
def predict_form():
    selected = []
    result = None
    if request.method == 'POST':
        selected = request.form.getlist('symptoms')
        result = predict_disease(selected)
    return render_template_string(
        BASE_TEMPLATE,
        symptoms=FEATURE_NAMES,
        selected=selected,
        result=result,
        meta=artifacts['meta'],
        feature_count=len(FEATURE_NAMES)
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    symptoms = data.get('symptoms', [])
    if not isinstance(symptoms, list):
        return jsonify({'error': 'symptoms must be a list of strings'}), 400
    result = predict_disease(symptoms)
    return jsonify(result)

@app.route('/api/symptoms', methods=['GET'])
def api_symptoms():
    return jsonify({'symptoms': FEATURE_NAMES})

@app.route('/api/metadata', methods=['GET'])
def api_metadata():
    meta = artifacts['meta']
    skinny = {
        'model_timestamp': meta.get('created'),
        'n_classes': len(LABEL_ENCODER.classes_),
        'n_features': len(FEATURE_NAMES),
        'classes': list(LABEL_ENCODER.classes_),  # Convert numpy array to list
        'params': meta.get('params', {})
    }
    return jsonify(skinny)

@app.route('/api/disease-info', methods=['GET'])
def api_disease_info():
    """Get disease descriptions and precautions"""
    return jsonify(disease_info)

@app.route('/api/disease-info/<disease>', methods=['GET'])
def api_disease_info_specific(disease):
    """Get specific disease information"""
    result = {}
    if disease in disease_info['descriptions']:
        result['description'] = disease_info['descriptions'][disease]
    if disease in disease_info['precautions']:
        result['precautions'] = disease_info['precautions'][disease]
    
    if not result:
        return jsonify({'error': f'Disease "{disease}" not found'}), 404
    
    result['disease'] = disease
    return jsonify(result)

@app.route('/api/ai/image-analysis', methods=['POST'])
def api_ai_image_analysis():
    """Analyze medical image with AI."""
    if not AI_FEATURES_AVAILABLE:
        return jsonify({'error': 'AI features not available'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    query = request.form.get('query', 
        """You are a professional doctor. What's in this image? Do you find anything wrong with it medically? 
        If you make a differential, suggest some remedies. Your response should be in one paragraph. 
        Answer as if you are answering to a real person. Don't say 'In the image I see' but say 'With what I see, I think you have....'
        Keep your answer concise (max 2 sentences). No preamble, start your answer right away.""")
    
    try:
        image_bytes = image_file.read()
        encoded_image = encode_image_from_bytes(image_bytes)
        analysis = analyze_image_with_groq(query, encoded_image)
        
        return jsonify({
            'analysis': analysis,
            'query': query,
            'filename': image_file.filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/transcribe', methods=['POST'])
def api_ai_transcribe():
    """Transcribe audio to text."""
    if not AI_FEATURES_AVAILABLE:
        return jsonify({'error': 'AI features not available'}), 503
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    try:
        audio_bytes = audio_file.read()
        transcription = transcribe_audio_with_groq(audio_bytes)
        
        return jsonify({
            'transcription': transcription,
            'filename': audio_file.filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/text-to-speech', methods=['POST'])
def api_ai_text_to_speech():
    """Convert text to speech."""
    if not AI_FEATURES_AVAILABLE:
        return jsonify({'error': 'AI features not available'}), 503
    
    data = request.get_json(force=True, silent=True) or {}
    text = data.get('text', '')
    language = data.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        audio_bytes = text_to_speech_gtts(text, language)
        if audio_bytes:
            # Return base64 encoded audio
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            return jsonify({
                'audio': audio_base64,
                'text': text,
                'language': language
            })
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/full-consultation', methods=['POST'])
def api_ai_full_consultation():
    """Complete AI consultation with audio transcription, image analysis, and TTS response."""
    if not AI_FEATURES_AVAILABLE:
        return jsonify({'error': 'AI features not available'}), 503
    
    # Get transcription from audio if provided
    transcription = ""
    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        transcription = transcribe_audio_with_groq(audio_bytes)
    
    # Get image analysis if provided
    image_analysis = ""
    if 'image' in request.files:
        image_file = request.files['image']
        image_bytes = image_file.read()
        encoded_image = encode_image_from_bytes(image_bytes)
        
        # Combine transcription with medical analysis prompt
        query = f"""You are a professional doctor. {transcription} What's in this image? 
        Do you find anything wrong with it medically? If you make a differential, suggest some remedies. 
        Your response should be in one paragraph. Answer as if you are answering to a real person. 
        Don't say 'In the image I see' but say 'With what I see, I think you have....'
        Keep your answer concise (max 2 sentences). No preamble, start your answer right away."""
        
        image_analysis = analyze_image_with_groq(query, encoded_image)
    
    # Generate audio response
    response_text = image_analysis if image_analysis else "I need more information to provide a medical consultation."
    audio_bytes = text_to_speech_gtts(response_text)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8') if audio_bytes else None
    
    return jsonify({
        'transcription': transcription,
        'analysis': image_analysis,
        'response_text': response_text,
        'response_audio': audio_base64
    })

@app.route('/health')
def health():
    return {'status': 'ok', 'loaded_features': len(FEATURE_NAMES), 'ai_available': AI_FEATURES_AVAILABLE}

# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
