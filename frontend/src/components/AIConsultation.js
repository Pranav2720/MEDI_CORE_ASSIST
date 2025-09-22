import React, { useState, useRef } from 'react';
import axios from 'axios';

export default function AIConsultation() {
  const [audioFile, setAudioFile] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [imageAnalysis, setImageAnalysis] = useState('');
  const [responseAudio, setResponseAudio] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
        setAudioFile(audioFile);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      setError('Error accessing microphone: ' + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageFile(file);
    }
  };

  const handleAudioUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setAudioFile(file);
    }
  };

  const handleFullConsultation = async () => {
    if (!audioFile && !imageFile) {
      setError('Please provide either an audio recording or an image for consultation.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      if (audioFile) formData.append('audio', audioFile);
      if (imageFile) formData.append('image', imageFile);

      const response = await axios.post('http://127.0.0.1:5000/api/ai/full-consultation', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = response.data;
      setTranscription(data.transcription);
      setImageAnalysis(data.analysis);
      
      if (data.response_audio) {
        const audioBlob = new Blob([Uint8Array.from(atob(data.response_audio), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setResponseAudio(audioUrl);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to process consultation');
    } finally {
      setLoading(false);
    }
  };

  const handleTextToSpeech = async (text) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/ai/text-to-speech', {
        text: text,
        language: 'en'
      });

      if (response.data.audio) {
        const audioBlob = new Blob([Uint8Array.from(atob(response.data.audio), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
      }
    } catch (err) {
      console.error('Text-to-speech error:', err);
    }
  };

  const playResponseAudio = () => {
    if (responseAudio && audioRef.current) {
      audioRef.current.play();
    }
  };

  return (
    <div className="ai-consultation">
      <div className="consultation-header">
        <h2>🤖 AI Doctor Consultation</h2>
        <p>Upload an image and/or record your voice for AI medical analysis</p>
      </div>

      {error && (
        <div className="error-message">
          <span>❌ {error}</span>
        </div>
      )}

      <div className="consultation-inputs">
        <div className="audio-section">
          <h3>🎤 Voice Input</h3>
          
          <div className="recording-controls">
            <button 
              onClick={isRecording ? stopRecording : startRecording}
              className={`record-btn ${isRecording ? 'recording' : ''}`}
              disabled={loading}
            >
              {isRecording ? '⏹️ Stop Recording' : '🎤 Start Recording'}
            </button>
            
            <div className="file-upload">
              <label htmlFor="audio-upload" className="upload-label">
                📁 Upload Audio File
              </label>
              <input
                id="audio-upload"
                type="file"
                accept="audio/*"
                onChange={handleAudioUpload}
                disabled={loading}
              />
            </div>
          </div>

          {audioFile && (
            <div className="file-info">
              <span>📄 Audio: {audioFile.name}</span>
            </div>
          )}
        </div>

        <div className="image-section">
          <h3>🖼️ Medical Image</h3>
          
          <div className="file-upload">
            <label htmlFor="image-upload" className="upload-label">
              📁 Upload Medical Image
            </label>
            <input
              id="image-upload"
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              disabled={loading}
            />
          </div>

          {imageFile && (
            <div className="image-preview">
              <img 
                src={URL.createObjectURL(imageFile)} 
                alt="Medical image preview" 
                style={{ maxWidth: '200px', maxHeight: '200px', objectFit: 'contain' }}
              />
              <span>📄 Image: {imageFile.name}</span>
            </div>
          )}
        </div>
      </div>

      <div className="consultation-action">
        <button 
          onClick={handleFullConsultation}
          disabled={loading || (!audioFile && !imageFile)}
          className="consult-btn"
        >
          {loading ? '🔄 Processing...' : '🩺 Get AI Consultation'}
        </button>
      </div>

      {(transcription || imageAnalysis) && (
        <div className="consultation-results">
          {transcription && (
            <div className="transcription-result">
              <h3>📝 Your Question:</h3>
              <p>{transcription}</p>
              <button onClick={() => handleTextToSpeech(transcription)} className="tts-btn">
                🔊 Play
              </button>
            </div>
          )}

          {imageAnalysis && (
            <div className="analysis-result">
              <h3>🩺 AI Doctor's Analysis:</h3>
              <p>{imageAnalysis}</p>
              <button onClick={() => handleTextToSpeech(imageAnalysis)} className="tts-btn">
                🔊 Play Response
              </button>
            </div>
          )}

          {responseAudio && (
            <div className="audio-response">
              <h3>🔊 Audio Response:</h3>
              <audio ref={audioRef} src={responseAudio} controls />
              <button onClick={playResponseAudio} className="play-btn">
                ▶️ Play Doctor's Response
              </button>
            </div>
          )}
        </div>
      )}

      <div className="disclaimer">
        <p>
          <strong>⚠️ Medical Disclaimer:</strong> This AI consultation is for educational purposes only. 
          Always consult with qualified healthcare professionals for proper medical diagnosis and treatment.
        </p>
      </div>
    </div>
  );
}