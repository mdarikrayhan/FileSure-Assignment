import re
import whisper

import noisereduce as nr
import librosa
import soundfile as sf

import librosa
import numpy as np

def reduce_noise(input_audio, output_audio):
    # Load audio file
    audio, sr = librosa.load(input_audio, sr=None)
    
    # Reduce noise
    reduced_audio = nr.reduce_noise(y=audio, sr=sr)
    
    # Save the cleaned audio
    sf.write(output_audio, reduced_audio, sr)

    return output_audio

def normalize_and_trim(input_audio, output_audio):
    # Load audio
    audio, sr = librosa.load(input_audio, sr=None)
    
    # Normalize audio volume
    audio = librosa.util.normalize(audio)
    
    # Trim silence
    trimmed_audio, _ = librosa.effects.trim(audio)
    
    # Save the processed audio
    sf.write(output_audio, trimmed_audio, sr)
    return output_audio


# Load Whisper model
model = whisper.load_model("turbo")

# Preprocess the audio: Noise reduction and normalization
cleaned_audio = reduce_noise("captcha_0002.wav", "cleaned_captcha.wav")
final_audio = normalize_and_trim(cleaned_audio, "final_captcha.wav")

# Transcribe the audio captcha
result = model.transcribe(final_audio, fp16=False)

# Extract letters before commas and periods
captcha_text = "".join(re.findall(r'(\w)(?=[,.])', result["text"]))

#print("Transcribed Text:",result["text"])
print("Formatted Captcha:", captcha_text)
