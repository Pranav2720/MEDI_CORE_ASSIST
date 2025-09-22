import React from 'react';

export default function PredictionResult({ data }) {
  return (
    <div className="prediction-result">
      <div className="result-header">
        <h2>🎯 Prediction Result</h2>
      </div>

      <div className="primary-prediction">
        <div className="disease-name">
          <span className="label">Predicted Disease:</span>
          <span className="disease">{data.predicted_disease}</span>
        </div>
        
        {/* Disease Description */}
        {data.description && (
          <div className="disease-description">
            <h3>📖 About this condition:</h3>
            <p>{data.description}</p>
          </div>
        )}

        {/* Precautions */}
        {data.precautions && data.precautions.length > 0 && (
          <div className="precautions-section">
            <h3>⚠️ Recommended Precautions:</h3>
            <ul className="precautions-list">
              {data.precautions.map((precaution, index) => (
                <li key={index} className="precaution-item">
                  {precaution}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {data.top_predictions && data.top_predictions.length > 0 && (
        <div className="probability-section">
          <h3>📊 Alternative Possibilities</h3>
          <div className="probability-list">
            {data.top_predictions.slice(1, 4).map((row, index) => (
              <div key={row.disease} className="probability-item">
                <div className="rank">#{index + 2}</div>
                <div className="disease-info">
                  <span className="disease-name">{row.disease}</span>
                  <div className="probability-bar">
                    <div 
                      className="probability-fill" 
                      style={{ width: `${row.probability * 100}%` }}
                    ></div>
                    <span className="probability-text">
                      {(row.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  {/* Show description for alternative predictions if available */}
                  {row.description && (
                    <div className="alt-description">
                      <small>{row.description}</small>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="input-summary">
        <h3>📝 Based on these symptoms:</h3>
        <div className="symptom-tags">
          {data.input_symptoms.map(symptom => (
            <span key={symptom} className="symptom-tag">
              {symptom.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </div>

      <div className="disclaimer-box">
        <p>
          <strong>Important:</strong> This is an AI prediction for educational purposes only. 
          Please consult with qualified healthcare professionals for proper medical diagnosis and treatment.
        </p>
      </div>
    </div>
  );
}
