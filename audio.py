from abc import ABC, abstractmethod
import pyaudio
from six.moves import queue

class IAudioRecorder(ABC):
    @abstractmethod
    def generator(self):
        pass

class IAudioPlayer(ABC):
    @abstractmethod
    def play(self, data: bytes):
        pass


class MicrophoneStream(IAudioRecorder):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate: int, chunk: int):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self._audio_interface = None
        self._audio_stream = None

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        if self._audio_interface:
            self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b''.join(data)


class AudioPlayer(IAudioPlayer):
    """Handles raw audio playback via PyAudio."""
    def __init__(self, rate: int, chunk: int, channels: int = 2, output_device_index: int = None):
        self._rate = rate
        self._chunk = chunk
        self._channels = channels
        self._output_device_index = output_device_index

    def play(self, data: bytes):
        if not data:
            return
        p = pyaudio.PyAudio()
        try:
            kwargs = {
                'format': pyaudio.paInt16,
                'channels': self._channels,
                'rate': self._rate,
                'output': True,
                'frames_per_buffer': self._chunk
            }
            if self._output_device_index is not None:
                kwargs['output_device_index'] = self._output_device_index

            stream = p.open(**kwargs)
            stream.write(data)
            stream.close()
        finally:
            p.terminate()
