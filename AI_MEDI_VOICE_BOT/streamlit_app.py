# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# VoiceBot UI with Streamlit
import os
import streamlit as st
import tempfile
import base64
from pathlib import Path

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

# load_dotenv()

st.set_page_config(page_title="AI Doctor with Vision and Voice", layout="wide")
st.title("AI Doctor with Vision and Voice")

system_prompt="""You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""

# Create a two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Input")
    
    # Audio upload instead of recording (since st.audio_recorder is not available)
    st.write("Please upload an audio file with your question:")
    audio_file = st.file_uploader("Upload your question (audio file)", type=["wav", "mp3", "m4a"])
    
    # Image upload
    uploaded_image = st.file_uploader("Upload an image for diagnosis", type=["png", "jpg", "jpeg"])
    
    # Process button
    process_button = st.button("Process")

with col2:
    st.subheader("Doctor's Response")
    
    # Placeholders for outputs
    speech_to_text_output_container = st.empty()
    doctor_response_container = st.empty()
    voice_output_container = st.empty()

if process_button and (audio_file or uploaded_image):
    with st.spinner("Processing your request..."):
        # Save audio to a temporary file if provided
        audio_filepath = None
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_audio:
                tmp_audio.write(audio_file.getbuffer())
                audio_filepath = tmp_audio.name
        
        # Save image to a temporary file if provided
        image_filepath = None
        if uploaded_image:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_image:
                tmp_image.write(uploaded_image.getbuffer())
                image_filepath = tmp_image.name
        
        # Process the inputs
        if audio_filepath:
            speech_to_text_output = transcribe_with_groq(
                GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                audio_filepath=audio_filepath,
                stt_model="whisper-large-v3"
            )
            speech_to_text_output_container.text_area("Your question:", speech_to_text_output, height=100)
        else:
            speech_to_text_output = ""
            speech_to_text_output_container.warning("No audio uploaded. Please upload an audio file with your question.")
        
        # Handle the image input
        if image_filepath:
            doctor_response = analyze_image_with_query(
                query=system_prompt + speech_to_text_output, 
                encoded_image=encode_image(image_filepath), 
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            doctor_response_container.text_area("Doctor's analysis:", doctor_response, height=200)
            
            # Generate audio response
            output_audio_path = "final.mp3"
            text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath=output_audio_path)
            
            # Display audio player
            if os.path.exists(output_audio_path):
                with open(output_audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    voice_output_container.audio(audio_bytes, format="audio/mp3")
            else:
                voice_output_container.warning("Failed to generate audio response.")
        else:
            doctor_response_container.warning("No image provided for analysis. Please upload an image.")
        
        # Clean up temporary files
        if audio_filepath and os.path.exists(audio_filepath):
            os.remove(audio_filepath)
        if image_filepath and os.path.exists(image_filepath):
            os.remove(image_filepath)
        if os.path.exists("final.mp3"):
            # We don't delete the audio file right away so it can be played by the user
            pass

# Alternative method to record audio using HTML and JavaScript
st.markdown("---")
st.subheader("Alternative: Record Audio Directly")
st.markdown("""
<script>
let audioChunks = [];
let mediaRecorder;
let recording = false;
let audioBlob;

function startRecording() {
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;
    document.getElementById('status').innerText = 'Recording...';
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = e => {
                audioChunks.push(e.data);
            };
            mediaRecorder.onstop = () => {
                audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                document.getElementById('audio-player').src = audioUrl;
                document.getElementById('audio-player').style.display = 'block';
                document.getElementById('download-btn').disabled = false;
                
                // Convert to base64 for download
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    const base64data = reader.result;
                    document.getElementById('download-btn').href = base64data;
                };
            };
            
            audioChunks = [];
            mediaRecorder.start();
            recording = true;
        });
}

function stopRecording() {
    if (recording && mediaRecorder) {
        mediaRecorder.stop();
        recording = false;
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('status').innerText = 'Recording stopped. Upload the downloaded file above.';
    }
}
</script>

<div>
    <button id="start-btn" onclick="startRecording()">Start Recording</button>
    <button id="stop-btn" onclick="stopRecording()" disabled>Stop Recording</button>
    <p id="status">Click Start Recording to begin</p>
    <audio id="audio-player" controls style="display:none;"></audio>
    <a id="download-btn" download="recorded_question.wav" disabled style="display:block; margin-top: 10px;">Download Recorded Audio</a>
</div>
""", unsafe_allow_html=True)

# Instructions at the bottom
st.markdown("---")
st.markdown("""
### How to use:
1. Option 1: Upload an audio file with your question
   - Or use the Alternative Recording section below, then download and upload the file
2. Upload an image related to your medical concern
3. Click "Process" to get the AI doctor's diagnosis and advice
4. Listen to the doctor's voice response

### Note:
- If using the alternative recording method, you'll need to download the file and then upload it in the section above.
- Make sure your browser allows microphone access if using the recording feature.
- For best results, speak clearly and ensure the image is well-lit and focused.
""")

# Install instructions for ffmpeg
st.markdown("---")
st.subheader("Troubleshooting")
st.markdown("""
If you see a warning about ffmpeg not being found, you need to install it:

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- **Mac**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

After installing, restart the Streamlit app.
""")