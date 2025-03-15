"""
Speech-to-Text System 


Features:
- Supports both Google Web Speech API (online) and Wav2Vec2 (offline)
- Handles common audio formats (WAV, MP3)
- Error handling for invalid files/inputs
"""

import argparse
import speech_recognition as sr
import librosa
from transformers import pipeline
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

def google_transcribe(audio_path):
    """Transcribe using Google's Web Speech API (requires internet)"""
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = r.record(source)
            return r.recognize_google(audio)
    except sr.UnknownValueError:
        raise Exception("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        raise Exception(f"Could not request results: {str(e)}")

def wav2vec_transcribe(audio_path):
    """Transcribe using Facebook's Wav2Vec2 model (works offline)"""
    try:
        # Load and resample audio to 16kHz
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Initialize ASR pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h"
        )
        
        # Process in chunks for longer audio
        return asr(speech, chunk_length_s=10, stride_length_s=(4, 2))["text"]
    except Exception as e:
        raise Exception(f"Wav2Vec transcription failed: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text System")
    parser.add_argument("audio_file", help="Path to audio file (WAV/MP3)")
    parser.add_argument("--model", choices=["google", "wav2vec"], default="wav2vec",
                       help="Choose recognition model (default: wav2vec)")
    
    args = parser.parse_args()

    print(f"\nProcessing {args.audio_file} with {args.model} model...\n")

    try:
        if args.model == "google":
            result = google_transcribe(args.audio_file)
        else:
            result = wav2vec_transcribe(args.audio_file)
        
        print("Transcription Result:")
        print("-" * 30)
        print(result)
        print("-" * 30)
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Tips:")
        print("- For Google model: Check internet connection and audio quality")
        print("- For Wav2Vec: Ensure audio is <30s and in WAV/MP3 format")

if __name__ == "__main__":
    main()
