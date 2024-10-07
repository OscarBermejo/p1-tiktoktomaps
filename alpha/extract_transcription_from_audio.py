import whisper
import os
import sys
import warnings 

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logger_config

# Suppress all warnings
warnings.filterwarnings("ignore")

logger = logger_config.get_logger(__name__)
def transcribe_audio(audio_path, model_name='base'): #choose model from 'tiny', 'base', 'small', 'medium', 'large'
    """
    Transcribes the given audio file using OpenAI's Whisper model.

    Args:
        audio_path (str): Path to the input WAV audio file.
        model_name (str): Name of the Whisper model to use. Options: tiny, base, small, medium, large.
    """
    # Check if the audio file exists
    if not os.path.isfile(audio_path):
        logger.info(f"Audio file not found: {audio_path}")
        return

    # Load the Whisper model
    logger.info(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    # Perform transcription
    logger.info(f"Transcribing audio file: {audio_path}")
    result = model.transcribe(audio_path)

    transcription = result['text'].strip()

    return transcription

def main(audio_path):
    return transcribe_audio(audio_path)

if __name__ == "__main__":
    main()