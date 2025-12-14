import vosk
import pyaudio
import json
import threading
from typing import Callable, Dict, Optional


class AudioKeywordDetector:
    """
    Real-time audio keyword detection using Vosk speech recognition.

    Listens to microphone input and triggers callbacks when specific keywords
    are detected in the transcribed speech.
    """

    DEFAULT_MODEL_PATH = "model/vosk-model-small-en-us-0.15"
    DEFAULT_RATE = 16000
    DEFAULT_CHUNK = 8000

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        rate: int = DEFAULT_RATE,
        chunk: int = DEFAULT_CHUNK,
        verbose: bool = True
    ):
        """
        Initialize the audio keyword detector.

        Args:
            model_path: Path to Vosk model directory
            rate: Audio sample rate in Hz
            chunk: Audio chunk size for processing
            verbose: Whether to print recognition results
        """
        self.model_path = model_path
        self.rate = rate
        self.chunk = chunk
        self.verbose = verbose

        self._model = None
        self._recognizer = None
        self._audio = None
        self._stream = None
        self._running = False
        self._thread = None
        self._callbacks: Dict[str, Callable] = {}

    def register_keyword(self, keyword: str, callback: Callable, aliases: list = None) -> None:
        """
        Register a callback function for a specific keyword.

        Args:
            keyword: The keyword to detect (case-insensitive)
            callback: Function to call when keyword is detected
            aliases: Optional list of alternative words that trigger the same callback
        """
        self._callbacks[keyword.lower()] = callback
        if aliases:
            for alias in aliases:
                self._callbacks[alias.lower()] = callback

    def unregister_keyword(self, keyword: str) -> None:
        """Remove a keyword callback."""
        self._callbacks.pop(keyword.lower(), None)

    def _initialize_audio(self) -> None:
        """Initialize Vosk model and PyAudio stream."""
        if self.verbose:
            print(f"Loading Vosk model from {self.model_path}...")

        self._model = vosk.Model(self.model_path)
        self._recognizer = vosk.KaldiRecognizer(self._model, self.rate)
        self._recognizer.SetWords(True)

        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        self._stream.start_stream()

        if self.verbose:
            print("Audio stream initialized")

    def _cleanup_audio(self) -> None:
        """Clean up audio resources."""
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()

    def _detect_keywords(self, text: str) -> None:
        """
        Check transcribed text for registered keywords and trigger callbacks.

        Args:
            text: Transcribed text to check for keywords
        """
        text_lower = text.lower()
        for keyword, callback in self._callbacks.items():
            if keyword in text_lower:
                if self.verbose:
                    print(f"Keyword '{keyword}' detected - triggering callback")
                try:
                    callback()
                except Exception as e:
                    print(f"Error executing callback for '{keyword}': {e}")

    def _listen_loop(self) -> None:
        """Main listening loop (runs in separate thread)."""
        try:
            while self._running:
                data = self._stream.read(self.chunk, exception_on_overflow=False)

                if self._recognizer.AcceptWaveform(data):
                    result = json.loads(self._recognizer.Result())
                    text = result.get('text', '')

                    if text:
                        if self.verbose:
                            print(f"Recognized: {text}")
                        self._detect_keywords(text)
                else:
                    partial = json.loads(self._recognizer.PartialResult())
                    partial_text = partial.get('partial', '')
                    if partial_text and self.verbose:
                        print(f"Partial: {partial_text}", end='\r')

        except Exception as e:
            print(f"Error in listening loop: {e}")
        finally:
            self._cleanup_audio()

    def start(self, blocking: bool = False) -> None:
        """
        Start listening for keywords.

        Args:
            blocking: If True, blocks until stop() is called.
                     If False, runs in background thread.
        """
        if self._running:
            print("Detector is already running")
            return

        self._initialize_audio()
        self._running = True

        if self.verbose:
            keywords = ", ".join(f"'{k}'" for k in self._callbacks.keys())
            print(f"Listening for keywords: {keywords}")
            print("Press Ctrl+C to stop\n")

        if blocking:
            try:
                self._listen_loop()
            except KeyboardInterrupt:
                if self.verbose:
                    print("\nStopping detector...")
                self.stop()
        else:
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop listening and clean up resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._cleanup_audio()
        if self.verbose:
            print("Detector stopped")

    def is_running(self) -> bool:
        """Check if detector is currently running."""
        return self._running


def main():
    """Demo usage of AudioKeywordDetector."""
    detector = AudioKeywordDetector()

    # Example callbacks with aliases
    detector.register_keyword("glove", lambda: print("Action: GLOVE"))
    detector.register_keyword("pliers", lambda: print("Action: PLIERS"), aliases=["player", "players", "playoffs"])
    detector.register_keyword("syringe", lambda: print("Action: SYRINGE"), aliases=["syrian", "surrender"])
    detector.register_keyword("stop", lambda: print("Action: STOP"), aliases=["step"])

    try:
        detector.start(blocking=True)
    except KeyboardInterrupt:
        detector.stop()


if __name__ == "__main__":
    main()