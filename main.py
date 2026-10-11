import sys
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech import enums
from google.cloud.speech_v1p1beta1 import types

from audio import MicrophoneStream, AudioPlayer
from translator_service import GoogleTranslationService, GoogleTextToSpeechService
from transcription_handler import TranscriptionHandler

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


def create_application():
    """Composition root initializing services and handlers."""
    audio_player = AudioPlayer(RATE, CHUNK)
    translation_service = GoogleTranslationService()
    tts_service = GoogleTextToSpeechService()
    handler = TranscriptionHandler(translation_service, tts_service, audio_player)
    return handler


def trans(handler: TranscriptionHandler):
    language_code = 'en-US'
    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code
    )
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)
        handler.handle(responses)


def main():
    handler = create_application()
    try:
        trans(handler)
    except KeyboardInterrupt:
        print('Interrupted by user. Exiting...')


if __name__ == '__main__':
    main()
