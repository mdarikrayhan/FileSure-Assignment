import whisper

model = whisper.load_model("turbo")
result = model.transcribe("captcha_0001.wav")
print(result["text"])