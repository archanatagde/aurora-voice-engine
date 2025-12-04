# assistant.py
# Upgraded voice assistant prototype with:
# - optional wake-word (Porcupine if installed) with fallback keyword detection
# - background listening (non-blocking)
# - async command processing
# - modular skill registration
# - pluggable TTS (pyttsx3 default, placeholder for cloud TTS)
# - simple NLU intent extractor (rule-based) with hooks for LLM/NLU services
# - basic context memory
#
# Requirements (optional features):
# pip install SpeechRecognition pyttsx3 wikipedia pyjokes pywhatkit vosk
# For Porcupine wake-word: pip install pvporcupine (and download model/keyword)
#
# This is a single-file runnable scaffold. Replace placeholders with production services as needed.

import queue
import threading
import time
import datetime
import json
import sys
import traceback
from typing import Optional, Dict, Any, Callable

try:
    import speech_recognition as sr
except Exception:
    raise RuntimeError("speech_recognition is required: pip install SpeechRecognition")

# Optional imports
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import wikipedia
except Exception:
    wikipedia = None

try:
    import pyjokes
except Exception:
    pyjokes = None

try:
    import pywhatkit
except Exception:
    pywhatkit = None

# Optional: porcupine wake word
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except Exception:
    pvporcupine = None
    PORCUPINE_AVAILABLE = False

# Optional: Vosk offline ASR for continuous transcription (advanced)
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    Model = None
    KaldiRecognizer = None
    VOSK_AVAILABLE = False

# ---------------------------
# Utilities and configuration
# ---------------------------

CONFIG = {
    "wake_word": "alexa",         # fallback keyword
    "porcupine_keyword_path": None,  # path to porcupine keyword file if used
    "porcupine_model_path": None,    # path to porcupine model file if used
    "use_porcupine": False,       # set True if pvporcupine configured
    "tts_engine": "pyttsx3",      # default tts
    "debug": True,
}

def debug_log(*args, **kwargs):
    if CONFIG.get("debug"):
        print("[DEBUG]", *args, **kwargs)

# ---------------------------
# Simple context memory
# ---------------------------

class ContextMemory:
    def _init_(self):
        self.short_term = {}  # ephemeral context for a session
        self.long_term = {}   # placeholder for persistent data (save to file if needed)

    def set(self, key, value, persistent=False):
        if persistent:
            self.long_term[key] = value
        else:
            self.short_term[key] = value

    def get(self, key, default=None):
        return self.short_term.get(key, self.long_term.get(key, default))

    def clear_short(self):
        self.short_term.clear()

memory = ContextMemory()

# ---------------------------
# TTS interface
# ---------------------------

class TTS:
    def _init_(self):
        self.impl = None
        if CONFIG["tts_engine"] == "pyttsx3" and pyttsx3 is not None:
            engine = pyttsx3.init()
            # choose voice if available (optional)
            try:
                voices = engine.getProperty("voices")
                # keep as default or pick second voice if available
                if len(voices) > 1:
                    engine.setProperty("voice", voices[1].id)
            except Exception:
                pass
            self.impl = engine
            self.say_method = self._pyttsx3_say
        else:
            # Placeholder for cloud TTS initialization (ElevenLabs/Azure/Polly)
            self.impl = None
            self.say_method = self._dummy_say

        self._speak_queue = queue.Queue()
        self._speaker_thread = threading.Thread(target=self._speaker_loop, daemon=True)
        self._speaker_thread.start()

    def _pyttsx3_say(self, text: str):
        try:
            self.impl.say(text)
            self.impl.runAndWait()
        except Exception:
            traceback.print_exc()

    def _dummy_say(self, text: str):
        # If no TTS installed, fallback to printing
        print("TTS:", text)

    def speak(self, text: str):
        # Non-blocking enqueue
        self._speak_queue.put(text)

    def _speaker_loop(self):
        while True:
            text = self._speak_queue.get()
            if text is None:
                break
            try:
                self.say_method(text)
            except Exception:
                traceback.print_exc()

tts = TTS()

# ---------------------------
# Wake-word detector
# ---------------------------

class WakeWordDetector:
    """
    Two modes:
     - Porcupine (if configured) -> very low false positive, offline
     - Fallback -> keyword presence in transcript (less reliable)
    """
    def _init_(self, wake_word: str = "alexa"):
        self.wake_word = wake_word.lower()
        self.use_porcupine = CONFIG.get("use_porcupine", False) and PORCUPINE_AVAILABLE
        self.porcupine = None
        if self.use_porcupine:
            try:
                # requires porcupine keyword and model path in CONFIG
                self.porcupine = pvporcupine.create(
                    keyword_paths=[CONFIG["porcupine_keyword_path"]],
                    model_path=CONFIG["porcupine_model_path"]
                )
            except Exception:
                debug_log("Porcupine initialization failed; falling back to transcript keyword detection")
                self.use_porcupine = False

    def audio_callback_detect(self, pcm):
        """
        If porcupine enabled, caller provides raw PCM frames and we return True on detection.
        """
        if not self.use_porcupine:
            return False
        try:
            result = self.porcupine.process(pcm)
            return result >= 0
        except Exception:
            return False

    def transcript_detect(self, transcript: str) -> bool:
        if not transcript:
            return False
        return self.wake_word in transcript.lower()

# ---------------------------
# Background listener (non-blocking)
# ---------------------------

class BackgroundListener:
    def _init_(self, phrase_time_limit: Optional[int] = 8):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.phrase_time_limit = phrase_time_limit
        self.audio_queue = queue.Queue()  # raw audio or transcripts
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self):
        with self.microphone as source:
            # calibrate ambient noise once
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            except Exception:
                pass
            while not self._stop_event.is_set():
                try:
                    debug_log("Listening for phrase...")
                    audio = self.recognizer.listen(source, phrase_time_limit=self.phrase_time_limit)
                    # enqueue raw audio object; main controller can transcribe or use other ASR
                    self.audio_queue.put(audio)
                except Exception as e:
                    debug_log("Listener error:", e)
                    time.sleep(0.5)

    def get_audio(self, timeout: Optional[float] = None):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()

# ---------------------------
# NLU (simple rule-based) + Intent extraction
# ---------------------------

def nlu_extract(transcript: str) -> Dict[str, Any]:
    """
    Very small rule-based NLU returning {"intent": str, "entities": {...}}
    Replace or augment with real NLU (spaCy, Rasa, LLM) for production.
    """
    text = (transcript or "").lower().strip()
    intent = "unknown"
    entities: Dict[str, Any] = {}

    # intent: play music
    if text.startswith("play "):
        intent = "play_music"
        entities["song"] = text.replace("play ", "", 1).strip()
        return {"intent": intent, "entities": entities}

    if "what time" in text or "time is it" in text or text == "time":
        intent = "get_time"
        return {"intent": intent, "entities": entities}

    if text.startswith("who is ") or text.startswith("who's ") or "who the heck is" in text:
        intent = "who_is"
        person = text.replace("who is", "").replace("who's", "").replace("who the heck is", "").strip()
        entities["person"] = person
        return {"intent": intent, "entities": entities}

    if "joke" in text:
        intent = "tell_joke"
        return {"intent": intent, "entities": entities}

    if "date" in text:
        intent = "date"
        return {"intent": intent, "entities": entities}

    # fallback: small talk
    if "are you single" in text:
        intent = "small_talk_single"
        return {"intent": intent, "entities": entities}

    # fallback unknown
    return {"intent": intent, "entities": entities}

# ---------------------------
# Skills / Command handlers
# ---------------------------

class SkillRegistry:
    def _init_(self):
        self.handlers: Dict[str, Callable[[Dict[str, Any]], None]] = {}

    def register(self, intent_name: str):
        def decorator(fn):
            self.handlers[intent_name] = fn
            return fn
        return decorator

    def handle(self, intent_name: str, entities: Dict[str, Any]):
        handler = self.handlers.get(intent_name)
        if handler:
            try:
                handler(entities)
            except Exception:
                traceback.print_exc()
                tts.speak("Sorry, I encountered an error handling that command.")
        else:
            tts.speak("I don't know how to do that yet.")

skills = SkillRegistry()

@skills.register("play_music")
def skill_play_music(entities):
    song = entities.get("song", "")
    if not song:
        tts.speak("What should I play?")
        return
    tts.speak(f"Playing {song}")
    # non-blocking: spawn thread to play via pywhatkit or other player
    def _play():
        if pywhatkit:
            try:
                pywhatkit.playonyt(song)
            except Exception:
                traceback.print_exc()
        else:
            debug_log("pywhatkit not installed; would play:", song)
    threading.Thread(target=_play, daemon=True).start()

@skills.register("get_time")
def skill_get_time(_entities):
    now = datetime.datetime.now().strftime("%I:%M %p")
    tts.speak(f"The time is {now}")

@skills.register("who_is")
def skill_who_is(entities):
    person = entities.get("person", "")
    if not person:
        tts.speak("Who do you want to know about?")
        return
    tts.speak(f"Searching Wikipedia for {person}")
    if wikipedia:
        try:
            summary = wikipedia.summary(person, sentences=1)
            memory.set("last_person", person)
            tts.speak(summary)
        except Exception:
            tts.speak("I couldn't find information about that person.")
    else:
        tts.speak("Wikipedia support is not installed.")

@skills.register("tell_joke")
def skill_tell_joke(_entities):
    if pyjokes:
        try:
            joke = pyjokes.get_joke()
            tts.speak(joke)
        except Exception:
            traceback.print_exc()
            tts.speak("I couldn't get a joke right now.")
    else:
        tts.speak("Joke module not available.")

@skills.register("date")
def skill_date(_entities):
    tts.speak("I am not available for dates. I am your assistant.")

@skills.register("small_talk_single")
def skill_small_talk_single(_entities):
    tts.speak("I am in a relationship with Wi Fi.")

# Default fallback
@skills.register("unknown")
def skill_unknown(_entities):
    tts.speak("Sorry, I didn't understand that. Please try again.")

# ---------------------------
# Main controller
# ---------------------------

class AssistantController:
    def _init_(self):
        self.listener = BackgroundListener()
        self.wake_detector = WakeWordDetector(wake_word=CONFIG["wake_word"])
        self.command_queue = queue.Queue()
        self._processing_thread = threading.Thread(target=self._command_processor_loop, daemon=True)
        self._processing_thread.start()
        self._stop_event = threading.Event()

    def _transcribe_audio(self, audio: sr.AudioData) -> Optional[str]:
        # Use Google recognizer as default (requires internet).
        # For offline alternative, integrate Vosk.
        recognizer = self.listener.recognizer
        try:
            text = recognizer.recognize_google(audio)
            debug_log("Transcribed:", text)
            return text
        except sr.UnknownValueError:
            debug_log("Could not understand audio")
            return None
        except sr.RequestError:
            debug_log("Speech recognition service request error")
            return None
        except Exception:
            traceback.print_exc()
            return None

    def _command_processor_loop(self):
        while not self._stop_event.is_set():
            try:
                intent_name, entities = self.command_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                skills.handle(intent_name, entities)
            except Exception:
                traceback.print_exc()

    def run(self):
        tts.speak("Assistant starting up")
        while True:
            # Get audio from background listener
            audio = self.listener.get_audio(timeout=1.0)
            if audio is None:
                continue

            # First pass transcription (fast)
            transcript = self._transcribe_audio(audio)
            if not transcript:
                continue

            # Check wake-word using wake detector transcript method (fallback)
            if self.wake_detector.transcript_detect(transcript):
                # remove wake word from transcript for command analysis
                cleaned = transcript.lower().replace(CONFIG["wake_word"], "").strip()
                if not cleaned:
                    # if only wake-word said, ask for command
                    tts.speak("Yes?")
                    # fetch next phrase
                    next_audio = self.listener.get_audio(timeout=5.0)
                    if next_audio:
                        next_transcript = self._transcribe_audio(next_audio)
                        transcript = next_transcript or ""
                    else:
                        transcript = ""
                else:
                    transcript = cleaned

                if not transcript:
                    continue

                # Extract intent
                nlu = nlu_extract(transcript)
                intent = nlu.get("intent", "unknown")
                entities = nlu.get("entities", {})
                debug_log("Detected intent:", intent, "entities:", entities)
                # Enqueue for async handling
                self.command_queue.put((intent, entities))
            else:
                # optional: you could implement "hotword via audio stream" using porcupine or VOSK
                debug_log("Wake word not found in:", transcript)
                # Optionally respond to short direct commands without wakeword if you want:
                # small_command_threshold = True
                # For now ignore non-wake transcripts
                continue

    def stop(self):
        self._stop_event.set()
        self.listener.stop()
        self.command_queue.put((None, None))
        tts.speak("Assistant shutting down")

# ---------------------------
# Entrypoint
# ---------------------------

if _name_ == "_main_":
    try:
        assistant = AssistantController()
        assistant.run()
    except KeyboardInterrupt:
        print("Exiting on user interrupt")
        try:
            assistant.stop()
        except Exception:
            pass
        sys.exit(0)
