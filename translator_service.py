from abc import ABC, abstractmethod
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech

class ITranslationService(ABC):
    @abstractmethod
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        pass

class ITextToSpeechService(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice_params, audio_config) -> bytes:
        pass


class GoogleTranslationService(ITranslationService):
    """Google Cloud Translation API wrapper."""
    def __init__(self):
        self._client = translate.Client()

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        result = self._client.translate(text, source_language=source_lang, target_language=target_lang)
        return result['translatedText']


class GoogleTextToSpeechService(ITextToSpeechService):
    """Google Cloud Text-To-Speech API wrapper."""
    def __init__(self, client=None, voice=None, audio_config=None):
        self._client = client or texttospeech.TextToSpeechClient()
        self._voice = voice or texttospeech.types.VoiceSelectionParams(
            language_code='hi-IN',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL
        )
        self._audio_config = audio_config or texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16
        )

    def synthesize(self, text: str, voice_params=None, audio_config=None) -> bytes:
        v = voice_params or self._voice
        cfg = audio_config or self._audio_config
        synthesis_input = texttospeech.types.SynthesisInput(text=text)
        response = self._client.synthesize_speech(synthesis_input, v, cfg)
        return response.audio_content
