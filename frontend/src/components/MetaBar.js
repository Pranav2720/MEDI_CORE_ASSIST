import React from 'react';

export default function MetaBar({ meta }) {
  if (!meta) return null;
  
  return (
    <div className="meta-bar">
      <div className="meta-item">
        <span className="meta-label">🤖 Model:</span>
        <span className="meta-value">Decision Tree</span>
      </div>
      <div className="meta-item">
        <span className="meta-label">📅 Trained:</span>
        <span className="meta-value">{meta.model_timestamp || 'Unknown'}</span>
      </div>
      <div className="meta-item">
        <span className="meta-label">🏷️ Features:</span>
        <span className="meta-value">{meta.n_features}</span>
      </div>
      <div className="meta-item">
        <span className="meta-label">🎯 Diseases:</span>
        <span className="meta-value">{meta.n_classes}</span>
      </div>
    </div>
  );
}
