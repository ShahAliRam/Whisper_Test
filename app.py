import os
import numpy as np
import soundfile as sf
import torch
from flask import Flask, render_template, request, jsonify, send_file
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datetime import datetime
import librosa
import glob
import atexit
import shutil

app = Flask(__name__)

# Create directories for storing audio files
os.makedirs('recordings', exist_ok=True)
os.makedirs('audios', exist_ok=True)

# Session history storage
session_history = []

# Clean up recordings folder on startup
def cleanup_recordings():
    if os.path.exists('recordings'):
        for file in glob.glob('recordings/*'):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    session_history.clear()
    print("Recordings cleaned up on startup")

# Register cleanup on exit
atexit.register(cleanup_recordings)

# Initialize Whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Loading Whisper large-v3-turbo model...")
model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
    return_timestamps=True,
    generate_kwargs={"language": "urdu", "task": "transcribe"} 
)

print("Model loaded successfully!")

# Cleanup old recordings on startup
cleanup_recordings()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sample_audios', methods=['GET'])
def get_sample_audios():
    """Get list of sample audio files from audios folder"""
    try:
        sample_files = []
        audio_files = sorted(glob.glob('audios/*.wav'))[:20]  # Limit to first 20 files
        
        for filepath in audio_files:
            filename = os.path.basename(filepath)
            sample_files.append({
                'filename': filename,
                'path': f'/audio/sample/{filename}'
            })
        
        return jsonify({'samples': sample_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/session_history', methods=['GET'])
def get_session_history():
    """Get the session recording history"""
    return jsonify({'history': session_history})


@app.route('/audio/sample/<filename>', methods=['GET'])
def serve_sample_audio(filename):
    """Serve sample audio file"""
    try:
        filepath = os.path.join('audios', filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/wav')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/audio/recording/<filename>', methods=['GET'])
def serve_recording_audio(filename):
    """Serve recorded audio file"""
    try:
        filepath = os.path.join('recordings', filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/wav')
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe_sample/<filename>', methods=['POST'])
def transcribe_sample(filename):
    """Transcribe a sample audio file"""
    try:
        filepath = os.path.join('audios', filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Sample file not found'}), 404
        
        # Load and convert to 8kHz mono
        audio_data, sample_rate = librosa.load(filepath, sr=8000, mono=True)
        
        # Save as 8kHz WAV temporarily
        temp_filepath = os.path.join('recordings', f'temp_{filename}')
        sf.write(temp_filepath, audio_data, 8000)
        
        # Transcribe using Whisper
        result = pipe(temp_filepath)
        transcription = result['text']
        
        # Clean up temp file
        os.remove(temp_filepath)
        
        return jsonify({
            'transcription': transcription,
            'filename': filename,
            'sample_rate': 8000
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Save the original audio file
        original_ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'webm'
        original_filename = f'recording_{timestamp}.{original_ext}'
        original_filepath = os.path.join('recordings', original_filename)
        audio_file.save(original_filepath)
        
        # Convert to 8kHz mono WAV for Whisper
        wav_filename = f'recording_{timestamp}.wav'
        wav_filepath = os.path.join('recordings', wav_filename)
        
        # Load audio file and convert to 8kHz mono
        audio_data, sample_rate = librosa.load(original_filepath, sr=8000, mono=True)
        
        # Save as 8kHz mono WAV
        sf.write(wav_filepath, audio_data, 8000)
        
        # Remove original file if it's not WAV
        if original_ext.lower() != 'wav':
            os.remove(original_filepath)
        
        # Transcribe using Whisper
        result = pipe(wav_filepath)
        transcription = result['text']
        
        # Add to session history
        history_entry = {
            'filename': wav_filename,
            'transcription': transcription,
            'timestamp': timestamp,
            'path': f'/audio/recording/{wav_filename}',
            'sample_rate': 8000
        }
        session_history.insert(0, history_entry)  # Add to beginning
        
        return jsonify({
            'transcription': transcription,
            'filename': wav_filename,
            'sample_rate': 8000
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
