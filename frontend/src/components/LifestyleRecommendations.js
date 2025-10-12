import React, { useState } from 'react';
import axios from 'axios';

function LifestyleRecommendations() {
  const [userProfile, setUserProfile] = useState({
    age: '',
    gender: '',
    height_cm: '',
    weight_kg: '',
    activity_level: 1,
    symptoms: [],
    conditions: [],
    symptom_severity: 0
  });
  
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);

  const activityLevels = [
    { value: 0, label: 'Sedentary (Little to no exercise)', icon: '🛋️' },
    { value: 1, label: 'Moderate (Light exercise 1-3 days/week)', icon: '🚶' },
    { value: 2, label: 'Active (Exercise 3+ days/week)', icon: '🏃' }
  ];

  const severityLevels = [
    { value: 0, label: 'Mild', color: '#4CAF50' },
    { value: 1, label: 'Moderate', color: '#FF9800' },
    { value: 2, label: 'Severe', color: '#F44336' }
  ];

  const commonSymptoms = [
    'Fatigue', 'Headache', 'Back Pain', 'Anxiety', 'Sleep Issues',
    'Digestive Issues', 'Joint Pain', 'Stress', 'Dizziness', 'Chest Pain'
  ];

  const commonConditions = [
    'Hypertension', 'Diabetes', 'Heart Disease', 'Obesity', 'Arthritis',
    'Asthma', 'Depression', 'Anxiety Disorder', 'High Cholesterol', 'Osteoporosis'
  ];

  const handleInputChange = (field, value) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleArrayToggle = (field, item) => {
    setUserProfile(prev => ({
      ...prev,
      [field]: prev[field].includes(item)
        ? prev[field].filter(i => i !== item)
        : [...prev[field], item]
    }));
  };

  const calculateBMI = () => {
    const { height_cm, weight_kg } = userProfile;
    if (height_cm && weight_kg) {
      const heightM = height_cm / 100;
      const bmi = weight_kg / (heightM * heightM);
      return bmi.toFixed(1);
    }
    return null;
  };

  const getBMICategory = (bmi) => {
    if (bmi < 18.5) return { category: 'Underweight', color: '#2196F3' };
    if (bmi < 25) return { category: 'Normal', color: '#4CAF50' };
    if (bmi < 30) return { category: 'Overweight', color: '#FF9800' };
    return { category: 'Obese', color: '#F44336' };
  };

  const generateRecommendations = async () => {
    if (!userProfile.age || !userProfile.gender) {
      setError('Please fill in at least age and gender');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/lifestyle-recommendations', {
        ...userProfile,
        age: parseInt(userProfile.age),
        height_cm: userProfile.height_cm ? parseFloat(userProfile.height_cm) : null,
        weight_kg: userProfile.weight_kg ? parseFloat(userProfile.weight_kg) : null
      });

      setRecommendations(response.data.recommendations);
    } catch (err) {
      setError(`Failed to generate recommendations: ${err.response?.data?.error || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setUserProfile({
      age: '',
      gender: '',
      height_cm: '',
      weight_kg: '',
      activity_level: 1,
      symptoms: [],
      conditions: [],
      symptom_severity: 0
    });
    setRecommendations(null);
    setCurrentStep(1);
    setError(null);
  };

  const renderStep1 = () => (
    <div className="step-content">
      <h3>📝 Basic Information</h3>
      <div className="form-grid">
        <div className="form-group">
          <label>Age *</label>
          <input
            type="number"
            value={userProfile.age}
            onChange={(e) => handleInputChange('age', e.target.value)}
            placeholder="Enter your age"
            min="1"
            max="120"
          />
        </div>
        
        <div className="form-group">
          <label>Gender *</label>
          <select
            value={userProfile.gender}
            onChange={(e) => handleInputChange('gender', e.target.value)}
          >
            <option value="">Select gender</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div className="form-group">
          <label>Height (cm)</label>
          <input
            type="number"
            value={userProfile.height_cm}
            onChange={(e) => handleInputChange('height_cm', e.target.value)}
            placeholder="e.g., 170"
            min="50"
            max="250"
          />
        </div>

        <div className="form-group">
          <label>Weight (kg)</label>
          <input
            type="number"
            value={userProfile.weight_kg}
            onChange={(e) => handleInputChange('weight_kg', e.target.value)}
            placeholder="e.g., 70"
            min="20"
            max="300"
          />
        </div>
      </div>

      {userProfile.height_cm && userProfile.weight_kg && (
        <div className="bmi-display">
          <h4>BMI Calculator</h4>
          {(() => {
            const bmi = calculateBMI();
            const bmiInfo = getBMICategory(parseFloat(bmi));
            return (
              <div className="bmi-result">
                <span className="bmi-value">BMI: {bmi}</span>
                <span 
                  className="bmi-category" 
                  style={{ color: bmiInfo.color, fontWeight: 'bold' }}
                >
                  ({bmiInfo.category})
                </span>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );

  const renderStep2 = () => (
    <div className="step-content">
      <h3>🏃 Activity & Health Status</h3>
      
      <div className="form-group">
        <label>Activity Level</label>
        <div className="activity-options">
          {activityLevels.map(level => (
            <label key={level.value} className="activity-option">
              <input
                type="radio"
                name="activity_level"
                value={level.value}
                checked={userProfile.activity_level === level.value}
                onChange={(e) => handleInputChange('activity_level', parseInt(e.target.value))}
              />
              <span className="activity-icon">{level.icon}</span>
              <span className="activity-label">{level.label}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>Current Symptoms (select all that apply)</label>
        <div className="checkbox-grid">
          {commonSymptoms.map(symptom => (
            <label key={symptom} className="checkbox-item">
              <input
                type="checkbox"
                checked={userProfile.symptoms.includes(symptom)}
                onChange={() => handleArrayToggle('symptoms', symptom)}
              />
              <span>{symptom}</span>
            </label>
          ))}
        </div>
      </div>

      {userProfile.symptoms.length > 0 && (
        <div className="form-group">
          <label>Overall Symptom Severity</label>
          <div className="severity-options">
            {severityLevels.map(level => (
              <label key={level.value} className="severity-option">
                <input
                  type="radio"
                  name="symptom_severity"
                  value={level.value}
                  checked={userProfile.symptom_severity === level.value}
                  onChange={(e) => handleInputChange('symptom_severity', parseInt(e.target.value))}
                />
                <span 
                  className="severity-indicator" 
                  style={{ backgroundColor: level.color }}
                ></span>
                <span>{level.label}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderStep3 = () => (
    <div className="step-content">
      <h3>🏥 Medical Conditions</h3>
      
      <div className="form-group">
        <label>Pre-existing Conditions (select all that apply)</label>
        <div className="checkbox-grid">
          {commonConditions.map(condition => (
            <label key={condition} className="checkbox-item">
              <input
                type="checkbox"
                checked={userProfile.conditions.includes(condition)}
                onChange={() => handleArrayToggle('conditions', condition)}
              />
              <span>{condition}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="form-summary">
        <h4>📋 Summary</h4>
        <div className="summary-grid">
          <div className="summary-item">
            <strong>Age:</strong> {userProfile.age}
          </div>
          <div className="summary-item">
            <strong>Gender:</strong> {userProfile.gender}
          </div>
          <div className="summary-item">
            <strong>Activity Level:</strong> {activityLevels.find(l => l.value === userProfile.activity_level)?.label}
          </div>
          <div className="summary-item">
            <strong>Symptoms:</strong> {userProfile.symptoms.length > 0 ? userProfile.symptoms.join(', ') : 'None'}
          </div>
          <div className="summary-item">
            <strong>Conditions:</strong> {userProfile.conditions.length > 0 ? userProfile.conditions.join(', ') : 'None'}
          </div>
        </div>
      </div>
    </div>
  );

  const renderRecommendations = () => (
    <div className="recommendations-display">
      <h3>🎯 Your Personalized Lifestyle Recommendations</h3>
      
      <div className="recommendation-header">
        <div className="priority-badge" data-priority={recommendations.priority_level}>
          Priority: {recommendations.priority_level.toUpperCase()}
        </div>
        <div className="confidence-score">
          Confidence: {(recommendations.confidence * 100).toFixed(1)}%
        </div>
      </div>

      <div className="recommendation-sections">
        <div className="recommendation-section">
          <h4>🌟 General Health Guidelines</h4>
          <ul className="recommendation-list">
            {recommendations.general_recommendations?.map((rec, idx) => (
              <li key={idx} className="recommendation-item">{rec}</li>
            ))}
          </ul>
        </div>

        {recommendations.age_specific?.length > 0 && (
          <div className="recommendation-section">
            <h4>👤 Age-Specific Recommendations</h4>
            <ul className="recommendation-list">
              {recommendations.age_specific.map((rec, idx) => (
                <li key={idx} className="recommendation-item">{rec}</li>
              ))}
            </ul>
          </div>
        )}

        {recommendations.specific_recommendations?.length > 0 && (
          <div className="recommendation-section">
            <h4>🎯 Targeted Recommendations</h4>
            <ul className="recommendation-list">
              {recommendations.specific_recommendations.map((rec, idx) => (
                <li key={idx} className="recommendation-item priority-high">{rec}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="recommendation-actions">
        <button className="btn-secondary" onClick={resetForm}>
          🔄 Generate New Recommendations
        </button>
        <button 
          className="btn-primary" 
          onClick={() => window.print()}
        >
          🖨️ Print Recommendations
        </button>
      </div>
    </div>
  );

  return (
    <div className="lifestyle-recommendations">
      <div className="header">
        <h2>🌱 Lifestyle Recommendation Generator</h2>
        <p>Get personalized health and lifestyle recommendations based on your profile</p>
      </div>

      {error && (
        <div className="error-message">
          <span>⚠️</span>
          {error}
        </div>
      )}

      {!recommendations ? (
        <div className="recommendation-wizard">
          <div className="step-indicator">
            <div className={`step ${currentStep >= 1 ? 'active' : ''}`}>1</div>
            <div className={`step ${currentStep >= 2 ? 'active' : ''}`}>2</div>
            <div className={`step ${currentStep >= 3 ? 'active' : ''}`}>3</div>
          </div>

          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
          {currentStep === 3 && renderStep3()}

          <div className="step-navigation">
            {currentStep > 1 && (
              <button 
                className="btn-secondary" 
                onClick={() => setCurrentStep(currentStep - 1)}
              >
                ← Previous
              </button>
            )}
            
            {currentStep < 3 ? (
              <button 
                className="btn-primary" 
                onClick={() => setCurrentStep(currentStep + 1)}
                disabled={currentStep === 1 && (!userProfile.age || !userProfile.gender)}
              >
                Next →
              </button>
            ) : (
              <button 
                className="btn-primary generate-btn" 
                onClick={generateRecommendations}
                disabled={loading}
              >
                {loading ? '🔄 Generating...' : '✨ Generate Recommendations'}
              </button>
            )}
          </div>
        </div>
      ) : (
        renderRecommendations()
      )}
    </div>
  );
}

export default LifestyleRecommendations;