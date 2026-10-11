import unittest
from unittest.mock import MagicMock
from audio import IAudioPlayer
from translator_service import ITranslationService, ITextToSpeechService
from transcription_handler import TranscriptionHandler

class MockAudioPlayer(IAudioPlayer):
    def __init__(self):
        self.played_data = []

    def play(self, data: bytes):
        self.played_data.append(data)


class MockTranslationService(ITranslationService):
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        return f"Translated({text})"


class MockTTSService(ITextToSpeechService):
    def synthesize(self, text: str, voice_params=None, audio_config=None) -> bytes:
        return b"audio_bytes_for_" + text.encode('utf-8')


class MockAlternative:
    def __init__(self, transcript):
        self.transcript = transcript

class MockResult:
    def __init__(self, transcript, is_final=True):
        self.alternatives = [MockAlternative(transcript)]
        self.is_final = is_final

class MockResponse:
    def __init__(self, results):
        self.results = results


class TestTranscriptionHandler(unittest.TestCase):
    def setUp(self):
        self.audio_player = MockAudioPlayer()
        self.translation_service = MockTranslationService()
        self.tts_service = MockTTSService()
        self.handler = TranscriptionHandler(
            self.translation_service,
            self.tts_service,
            self.audio_player
        )

    def test_handle_normal_speech(self):
        responses = [MockResponse([MockResult("Hello world")])]
        exit_triggered = self.handler.handle(responses)
        self.assertFalse(exit_triggered)
        self.assertEqual(len(self.audio_player.played_data), 1)
        self.assertIn(b"audio_bytes_for_Translated(Hello world)", self.audio_player.played_data)

    def test_handle_exit_command(self):
        responses = [MockResponse([MockResult("exit")])]
        exit_triggered = self.handler.handle(responses)
        self.assertTrue(exit_triggered)

    def test_handle_interim_results_ignored(self):
        responses = [MockResponse([MockResult("Hello", is_final=False)])]
        exit_triggered = self.handler.handle(responses)
        self.assertFalse(exit_triggered)
        self.assertEqual(len(self.audio_player.played_data), 0)


if __name__ == '__main__':
    unittest.main()
