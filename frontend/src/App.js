import React, { useEffect, useState, useMemo } from 'react';
import axios from 'axios';
import SymptomSelector from './components/SymptomSelector';
import PredictionResult from './components/PredictionResult';
import MetaBar from './components/MetaBar';
import AIConsultation from './components/AIConsultation';

function App() {
  const [symptoms, setSymptoms] = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('prediction'); // New state for tabs

  useEffect(() => {
    async function fetchData() {
      try {
        const [symRes, metaRes] = await Promise.all([
          axios.get('http://127.0.0.1:5000/api/symptoms'),
          axios.get('http://127.0.0.1:5000/api/metadata')
        ]);
        setSymptoms(symRes.data.symptoms || []);
        setMeta(metaRes.data);
        setError(null);
      } catch (e) {
        setError('Failed to load initial data. Ensure backend is running.');
        console.error('Fetch error:', e);
      }
    }
    fetchData();
  }, []);

  const canPredict = useMemo(() => selected.length > 0, [selected]);

  async function handlePredict() {
    if (!canPredict) return;
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post('http://127.0.0.1:5000/api/predict', { symptoms: selected });
      setResult(res.data);
    } catch (e) {
      setError(e.response?.data?.error || 'Prediction failed. Please try again.');
      console.error('Prediction error:', e);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setSelected([]);
    setResult(null);
    setError(null);
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>🏥 Medical AI Assistant</h1>
          <p>Advanced disease prediction and AI consultation platform</p>
        </header>

        <MetaBar meta={meta} />

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={`tab-btn ${activeTab === 'prediction' ? 'active' : ''}`}
            onClick={() => setActiveTab('prediction')}
          >
            🔬 Disease Prediction
          </button>
          <button 
            className={`tab-btn ${activeTab === 'consultation' ? 'active' : ''}`}
            onClick={() => setActiveTab('consultation')}
          >
            🤖 AI Doctor Consultation
          </button>
        </div>
        
        {error && (
          <div className="error-message">
            <span>⚠️</span>
            {error}
          </div>
        )}

        {/* Tab Content */}
        {activeTab === 'prediction' && (
          <div className="main-content">
            <SymptomSelector
              symptoms={symptoms}
              selected={selected}
              setSelected={setSelected}
            />

            <div className="actions">
              <button 
                className="predict-btn" 
                disabled={!canPredict || loading} 
                onClick={handlePredict}
              >
                {loading ? '🔄 Analyzing...' : '🔍 Predict Disease'}
              </button>
              <button 
                className="clear-btn" 
                onClick={handleClear} 
                disabled={loading}
              >
                🗑️ Clear All
              </button>
            </div>

            {result && <PredictionResult data={result} />}
          </div>
        )}

        {activeTab === 'consultation' && (
          <div className="main-content">
            <AIConsultation />
          </div>
        )}

        <footer className="footer">
          <p>
            Powered by Machine Learning • Model Features: {meta?.n_features || 'Loading...'} • 
            Diseases: {meta?.n_classes || 'Loading...'}
          </p>
          <p className="disclaimer">
            ⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical advice.
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
