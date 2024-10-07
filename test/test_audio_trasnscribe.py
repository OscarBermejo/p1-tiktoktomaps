import whisper

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Transcribe the audio file
    result = model.transcribe(file_path)

    # Extract and return the transcription text
    transcription_text = result['text']
    return transcription_text

# Path to your audio file
audio_file_path = '/home/ec2-user/tiktok-extractor-v2/files/audio/7185551271389072682.wav'

# Transcribe the audio file
transcription = transcribe_audio(audio_file_path)

# Print the transcription
print("Transcription:")
print(transcription)