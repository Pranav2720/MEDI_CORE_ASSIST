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
# Lifestyle Recommendation Generator
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class LifestyleRecommendationEngine:
    def __init__(self):
        self.model = None
        self.recommendation_encoder = None
        self.age_groups = ['child', 'young_adult', 'adult', 'senior']
        self.lifestyle_factors = ['sedentary', 'moderate', 'active']
        self.risk_levels = ['low', 'medium', 'high']
        
        # Comprehensive recommendation database
        self.recommendation_db = {
            'general': {
                'hydration': "Drink at least 8-10 glasses of water daily to maintain proper hydration.",
                'sleep': "Aim for 7-9 hours of quality sleep each night for optimal health.",
                'exercise': "Engage in at least 150 minutes of moderate aerobic activity weekly.",
                'nutrition': "Follow a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.",
                'stress': "Practice stress management techniques like meditation, deep breathing, or yoga.",
                'checkup': "Schedule regular health checkups and screenings as recommended by your healthcare provider."
            },
            'condition_specific': {
                'hypertension': [
                    "Reduce sodium intake to less than 2,300mg per day (ideally 1,500mg).",
                    "Maintain a healthy weight through diet and exercise.",
                    "Limit alcohol consumption and avoid smoking.",
                    "Monitor blood pressure regularly at home."
                ],
                'diabetes': [
                    "Monitor blood glucose levels as directed by your healthcare provider.",
                    "Follow a consistent meal schedule with controlled carbohydrate intake.",
                    "Exercise regularly to help control blood sugar levels.",
                    "Take medications as prescribed and attend regular medical appointments."
                ],
                'heart_disease': [
                    "Follow a heart-healthy diet low in saturated fats and cholesterol.",
                    "Engage in cardiac rehabilitation if recommended.",
                    "Take prescribed medications consistently.",
                    "Monitor symptoms and seek immediate medical attention for chest pain."
                ],
                'obesity': [
                    "Create a sustainable calorie deficit through diet and exercise.",
                    "Focus on whole foods and portion control.",
                    "Incorporate both cardiovascular and strength training exercises.",
                    "Consider working with a registered dietitian for personalized guidance."
                ],
                'mental_health': [
                    "Maintain regular sleep patterns and good sleep hygiene.",
                    "Stay connected with supportive friends and family.",
                    "Consider counseling or therapy if symptoms persist.",
                    "Practice mindfulness, meditation, or other stress-reduction techniques."
                ]
            },
            'age_specific': {
                'child': [
                    "Ensure adequate nutrition for growth and development.",
                    "Encourage outdoor play and physical activity.",
                    "Maintain regular pediatric checkups and vaccinations.",
                    "Limit screen time and promote reading and creative activities."
                ],
                'young_adult': [
                    "Establish healthy lifestyle habits early.",
                    "Focus on building strong social connections.",
                    "Manage academic or career stress effectively.",
                    "Avoid risky behaviors like excessive alcohol consumption or smoking."
                ],
                'adult': [
                    "Balance work and personal life to reduce stress.",
                    "Begin regular health screenings (blood pressure, cholesterol, cancer screenings).",
                    "Maintain physical fitness to prevent age-related decline.",
                    "Plan for financial and health security in later years."
                ],
                'senior': [
                    "Stay physically active to maintain mobility and independence.",
                    "Keep mind active through learning and social engagement.",
                    "Follow up regularly with healthcare providers for chronic condition management.",
                    "Ensure home safety to prevent falls and injuries."
                ]
            },
            'symptom_based': {
                'fatigue': [
                    "Ensure adequate sleep and maintain a consistent sleep schedule.",
                    "Evaluate diet for nutritional deficiencies.",
                    "Gradually increase physical activity to boost energy levels.",
                    "Consider stress management techniques."
                ],
                'headache': [
                    "Stay hydrated throughout the day.",
                    "Maintain regular sleep patterns.",
                    "Identify and avoid potential triggers (stress, certain foods, bright lights).",
                    "Practice relaxation techniques."
                ],
                'back_pain': [
                    "Maintain good posture, especially when sitting for long periods.",
                    "Strengthen core muscles through targeted exercises.",
                    "Use proper body mechanics when lifting heavy objects.",
                    "Consider ergonomic improvements to your workspace."
                ],
                'anxiety': [
                    "Practice deep breathing exercises and mindfulness meditation.",
                    "Maintain regular exercise routine to reduce stress hormones.",
                    "Limit caffeine and alcohol intake.",
                    "Consider professional counseling if symptoms interfere with daily life."
                ],
                'digestive_issues': [
                    "Eat smaller, more frequent meals throughout the day.",
                    "Identify and avoid trigger foods.",
                    "Stay hydrated and include fiber-rich foods in your diet.",
                    "Manage stress, as it can worsen digestive symptoms."
                ]
            }
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model with synthetic training data."""
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: age_group, gender, has_chronic_condition, activity_level, bmi_category, symptom_severity
        features = []
        labels = []
        
        for _ in range(n_samples):
            age_group = np.random.randint(0, 4)  # 0-3 for child, young_adult, adult, senior
            gender = np.random.randint(0, 2)  # 0-1 for female, male
            has_chronic = np.random.randint(0, 2)  # 0-1 for no, yes
            activity_level = np.random.randint(0, 3)  # 0-2 for sedentary, moderate, active
            bmi_category = np.random.randint(0, 4)  # 0-3 for underweight, normal, overweight, obese
            symptom_severity = np.random.randint(0, 3)  # 0-2 for mild, moderate, severe
            
            features.append([age_group, gender, has_chronic, activity_level, bmi_category, symptom_severity])
            
            # Generate recommendation category based on features
            if has_chronic:
                if age_group >= 2:  # adult or senior
                    rec_category = 'chronic_management'
                else:
                    rec_category = 'preventive_care'
            elif activity_level == 0:  # sedentary
                rec_category = 'fitness_improvement'
            elif symptom_severity >= 1:  # moderate to severe symptoms
                rec_category = 'symptom_management'
            else:
                rec_category = 'wellness_maintenance'
            
            labels.append(rec_category)
        
        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(features, labels)
        
        # Create label encoder for recommendations
        self.recommendation_encoder = LabelEncoder()
        self.recommendation_encoder.fit(['chronic_management', 'preventive_care', 'fitness_improvement', 
                                       'symptom_management', 'wellness_maintenance'])
    
    def get_age_group(self, age):
        """Convert age to age group."""
        if age < 18:
            return 0  # child
        elif age < 30:
            return 1  # young_adult
        elif age < 65:
            return 2  # adult
        else:
            return 3  # senior
    
    def get_bmi_category(self, height_cm, weight_kg):
        """Calculate BMI category."""
        if height_cm and weight_kg:
            bmi = weight_kg / ((height_cm / 100) ** 2)
            if bmi < 18.5:
                return 0  # underweight
            elif bmi < 25:
                return 1  # normal
            elif bmi < 30:
                return 2  # overweight
            else:
                return 3  # obese
        return 1  # default to normal if no data
    
    def generate_recommendations(self, user_profile):
        """Generate personalized lifestyle recommendations."""
        try:
            # Extract features from user profile
            age = user_profile.get('age', 30)
            gender = 1 if user_profile.get('gender', '').lower() == 'male' else 0
            height = user_profile.get('height_cm', 170)
            weight = user_profile.get('weight_kg', 70)
            activity_level = user_profile.get('activity_level', 1)  # 0-2
            symptoms = user_profile.get('symptoms', [])
            conditions = user_profile.get('conditions', [])
            symptom_severity = user_profile.get('symptom_severity', 0)  # 0-2
            
            # Prepare features for ML model
            age_group = self.get_age_group(age)
            has_chronic = 1 if conditions else 0
            bmi_category = self.get_bmi_category(height, weight)
            
            features = [[age_group, gender, has_chronic, activity_level, bmi_category, symptom_severity]]
            
            # Get ML prediction
            prediction = self.model.predict(features)[0]
            prediction_proba = self.model.predict_proba(features)[0]
            
            # Generate comprehensive recommendations
            recommendations = {
                'category': prediction,
                'confidence': float(max(prediction_proba)),
                'general_recommendations': [],
                'specific_recommendations': [],
                'age_specific': [],
                'priority_level': 'medium'
            }
            
            # Add general recommendations
            recommendations['general_recommendations'] = [
                self.recommendation_db['general']['hydration'],
                self.recommendation_db['general']['sleep'],
                self.recommendation_db['general']['exercise'],
                self.recommendation_db['general']['nutrition']
            ]
            
            # Add age-specific recommendations
            age_group_name = self.age_groups[age_group]
            recommendations['age_specific'] = self.recommendation_db['age_specific'][age_group_name]
            
            # Add condition-specific recommendations
            if conditions:
                for condition in conditions:
                    condition_lower = condition.lower().replace(' ', '_')
                    if condition_lower in self.recommendation_db['condition_specific']:
                        recommendations['specific_recommendations'].extend(
                            self.recommendation_db['condition_specific'][condition_lower]
                        )
                recommendations['priority_level'] = 'high'
            
            # Add symptom-based recommendations
            if symptoms:
                for symptom in symptoms:
                    symptom_lower = symptom.lower().replace(' ', '_')
                    if symptom_lower in self.recommendation_db['symptom_based']:
                        recommendations['specific_recommendations'].extend(
                            self.recommendation_db['symptom_based'][symptom_lower]
                        )
            
            # Determine priority level
            if symptom_severity >= 2 or has_chronic:
                recommendations['priority_level'] = 'high'
            elif symptom_severity >= 1 or activity_level == 0:
                recommendations['priority_level'] = 'medium'
            else:
                recommendations['priority_level'] = 'low'
            
            # Remove duplicates while preserving order
            recommendations['specific_recommendations'] = list(dict.fromkeys(recommendations['specific_recommendations']))
            
            return recommendations
            
        except Exception as e:
            return {
                'error': f'Failed to generate recommendations: {str(e)}',
                'general_recommendations': [self.recommendation_db['general']['checkup']],
                'priority_level': 'medium'
            }

# Initialize the lifestyle recommendation engine
lifestyle_engine = LifestyleRecommendationEngine()

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

@app.route('/api/lifestyle-recommendations', methods=['POST'])
def api_lifestyle_recommendations():
    """Generate personalized lifestyle recommendations based on user profile."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract user profile from request
        user_profile = {
            'age': data.get('age', 30),
            'gender': data.get('gender', ''),
            'height_cm': data.get('height_cm'),
            'weight_kg': data.get('weight_kg'),
            'activity_level': data.get('activity_level', 1),  # 0=sedentary, 1=moderate, 2=active
            'symptoms': data.get('symptoms', []),
            'conditions': data.get('conditions', []),
            'symptom_severity': data.get('symptom_severity', 0)  # 0=mild, 1=moderate, 2=severe
        }
        
        # Generate recommendations
        recommendations = lifestyle_engine.generate_recommendations(user_profile)
        
        return jsonify({
            'success': True,
            'user_profile': user_profile,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate recommendations: {str(e)}'
        }), 500

@app.route('/health')
def health():
    return {'status': 'ok', 'loaded_features': len(FEATURE_NAMES), 'ai_available': AI_FEATURES_AVAILABLE}

# -----------------------------------------------------------------------------
# Main Entry
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
