import re
import whisper

# Load Whisper model
model = whisper.load_model("turbo")

# Transcribe the audio captcha
result = model.transcribe("captcha_0003.wav", fp16=False)

# Extract letters before commas and periods
captcha_text = "".join(re.findall(r'(\w)(?=[,.])', result["text"]))

print("Transcribed Text:",result["text"])
print("Formatted Captcha:", captcha_text)
