import re
from audio import IAudioPlayer
from translator_service import ITranslationService, ITextToSpeechService

class TranscriptionHandler:
    """Coordinates transcription processing, translation, TTS, and playback."""
    def __init__(
        self,
        translation_service: ITranslationService,
        tts_service: ITextToSpeechService,
        audio_player: IAudioPlayer,
        source_lang: str = 'en',
        target_lang: str = 'hi'
    ):
        self._translation_service = translation_service
        self._tts_service = tts_service
        self._audio_player = audio_player
        self._source_lang = source_lang
        self._target_lang = target_lang

    def handle(self, responses) -> bool:
        """
        Iterates over speech recognition responses, translates final transcripts,
        synthesizes speech, plays audio, and checks for exit commands.
        Returns True if exit was triggered, False otherwise.
        """
        print('Starting transcription processing loop...')
        num_chars_printed = 0

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:
                num_chars_printed = len(transcript)
            else:
                text = transcript + overwrite_chars
                print('You Spoke: ', text)

                # Translate text
                translated_text = self._translation_service.translate_text(
                    text, self._source_lang, self._target_lang
                )
                print('Translated: ', translated_text)

                # Synthesize Speech
                audio_content = self._tts_service.synthesize(translated_text)

                # Play Audio
                if audio_content:
                    self._audio_player.play(audio_content)

                # Check for exit condition
                if re.search(r'\b(exit|quit)\b', transcript, re.I):
                    print('Exiting recognition loop...')
                    return True

                num_chars_printed = 0

        return False
