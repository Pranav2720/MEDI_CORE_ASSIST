import React, { useState, useMemo } from 'react';

export default function SymptomSelector({ symptoms, selected, setSelected }) {
  const [filter, setFilter] = useState('');

  const filtered = useMemo(() => {
    if (!filter) return symptoms;
    return symptoms.filter(s => s.toLowerCase().includes(filter.toLowerCase()));
  }, [filter, symptoms]);

  function toggle(symptom) {
    if (selected.includes(symptom)) {
      setSelected(selected.filter(s => s !== symptom));
    } else {
      setSelected([...selected, symptom]);
    }
  }

  return (
    <div className="symptom-selector">
      <div className="selector-header">
        <h2>📋 Select Your Symptoms</h2>
        <div className="selected-count">
          {selected.length} symptom{selected.length !== 1 ? 's' : ''} selected
        </div>
      </div>
      
      <div className="search-box">
        <input
          type="text"
          placeholder="🔍 Search symptoms..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="search-input"
        />
      </div>

      <div className="symptom-grid">
        {filtered.map(sym => (
          <div 
            key={sym} 
            className={`symptom-card ${selected.includes(sym) ? 'selected' : ''}`} 
            onClick={() => toggle(sym)}
          >
            <div className="symptom-checkbox">
              {selected.includes(sym) ? '✅' : '⬜'}
            </div>
            <span className="symptom-text">{sym.replace(/_/g, ' ')}</span>
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="empty-state">
          <span>🔍</span>
          <p>No symptoms match your search</p>
        </div>
      )}

      {selected.length > 0 && (
        <div className="selected-symptoms">
          <h3>Selected Symptoms:</h3>
          <div className="selected-list">
            {selected.map(sym => (
              <span key={sym} className="selected-tag">
                {sym.replace(/_/g, ' ')}
                <button onClick={() => toggle(sym)}>×</button>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
