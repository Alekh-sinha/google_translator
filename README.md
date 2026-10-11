# Refactored Google Translator, TTS & Speech Recognition Architecture

This document outlines the architectural refactoring of the original monolithic `text_to_speech_3.py` script from `Alekh-sinha/google_translator` into a clean, modular structure strictly adhering to **SOLID principles** and design patterns.

---

## 🏛️ Architecture & Module Breakdown

The codebase has been refactored into four core modules:

1. **`audio.py`** (Audio Recording & Playback)
   - Adheres to **Single Responsibility Principle (SRP)** and **Interface Segregation Principle (ISP)**.
   - Defines abstract interfaces: `IAudioRecorder` and `IAudioPlayer`.
   - **`MicrophoneStream`**: Handles real-time microphone recording input via `pyaudio` with thread-safe buffering (`queue.Queue`) and context management (`__enter__`/`__exit__`).
   - **`AudioPlayer`**: Encapsulates audio output playback (`pyaudio` streaming write).

2. **`translator_service.py`** (External Translation & TTS APIs)
   - Adheres to **Single Responsibility Principle (SRP)** and **Dependency Inversion Principle (DIP)**.
   - Defines abstract interfaces: `ITranslationService` and `ITextToSpeechService`.
   - **`GoogleTranslationService`**: Encapsulates Google Cloud Translation API v2 (`translate_v2`).
   - **`GoogleTextToSpeechService`**: Encapsulates Google Cloud Text-to-Speech API (`texttospeech`).
   - Separates external SDK integration details from business logic, making unit testing straightforward through mocking.

3. **`transcription_handler.py`** (Business Logic & Response Processing Loop)
   - Adheres to **SRP** and **DIP**.
   - **`TranscriptionHandler`**: Manages the streaming server response generator loop, interim vs. final transcript parsing, delegation to translation and TTS services, audio playback triggering, and exit command detection (`exit` / `quit`).
   - Decoupled from concrete audio or API implementations via dependency injection.

4. **`main.py`** (Composition Root & Entry Point)
   - Adheres to **Dependency Inversion Principle (DIP)** via a composition root (`create_application()`).
   - Wires up concrete implementations to abstract interfaces, initializes the Google Cloud Speech client (`SpeechClient`), sets up streaming configurations (`StreamingRecognitionConfig`), and starts the recording/transcription pipeline.

---

## 📐 SOLID Principles Applied

- **S (Single Responsibility Principle)**: Each class has one clear responsibility. Audio recording (`MicrophoneStream`), playback (`AudioPlayer`), translation (`GoogleTranslationService`), text-to-speech synthesis (`GoogleTextToSpeechService`), and stream event handling (`TranscriptionHandler`) are fully separated into dedicated classes/modules.
- **O (Open/Closed Principle)**: New translation backends (e.g., DeepL, Azure Translator) or audio drivers can be added easily by implementing `ITranslationService` or `IAudioPlayer` without modifying existing business logic.
- **L (Liskov Substitution Principle)**: Any implementation of `IAudioRecorder`, `IAudioPlayer`, `ITranslationService`, or `ITextToSpeechService` can be substituted seamlessly in the application.
- **I (Interface Segregation Principle)**: Interfaces (`IAudioRecorder`, `IAudioPlayer`, `ITranslationService`, `ITextToSpeechService`) are small, focused, and client-specific.
- **D (Dependency Inversion Principle)**: High-level modules (`TranscriptionHandler`) depend on abstractions (`ITranslationService`, `ITextToSpeechService`, `IAudioPlayer`) rather than concrete SDK classes. Dependencies are injected via constructors.
