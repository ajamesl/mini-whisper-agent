"""Mini Whisper Agent - Voice command processor using OpenAI Whisper.

A voice-controlled assistant that listens for commands and executes actions
like Google searches and ChatGPT interactions.
"""

import re
import time
import webbrowser
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pyautogui
import sounddevice as sd
import torch
import whisper
from pynput import keyboard

# Constants
SAMPLE_RATE = 16000
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "whisper_tiny_finetuned.pt"
GOOGLE_URL = "https://www.google.com"
CHATGPT_URL = "https://chatgpt.com/?temporary-chat=true" # Remove the temporary chat parameter if not needed

# Command patterns
SEARCH_PATTERN = r"(?:search google(?: for)?|google for|search for)\s+(.+)"
CHATGPT_PATTERN = r"ask\s+chat\s*gpt(?:\s*to)?\s+(.+)|ask\s+chatgpt(?:\s*to)?\s+(.+)"

# Sleep durations - Change to your preference to match the speed of your system
BROWSER_LOAD_DELAY = 1.5    
CHATGPT_LOAD_DELAY = 2.0
TYPING_DELAY = 0.3
CHATGPT_TYPING_DELAY = 0.5


def execute_command(transcription: str) -> None:
    """Execute voice commands based on transcribed speech.

    Processes the transcribed text to identify and execute specific commands:
    - Google search commands: Opens Google and searches for specified terms
    - ChatGPT commands: Opens ChatGPT and sends specified prompts

    Args:
        transcription: The transcribed speech text to process.
    """
    text = transcription.lower().strip()

    # Handle Google search commands
    match_search = re.search(SEARCH_PATTERN, text, re.IGNORECASE)
    if match_search:
        query = match_search.group(1).strip()
        webbrowser.open(GOOGLE_URL)
        time.sleep(BROWSER_LOAD_DELAY)
        pyautogui.typewrite(query)
        time.sleep(TYPING_DELAY)
        pyautogui.press("enter")
        return

    # Handle ChatGPT commands
    match_chatgpt = re.search(CHATGPT_PATTERN, text, re.IGNORECASE)
    if match_chatgpt:
        prompt = match_chatgpt.group(1).strip()
        webbrowser.open(CHATGPT_URL)
        time.sleep(CHATGPT_LOAD_DELAY)
        pyautogui.typewrite(prompt)
        time.sleep(CHATGPT_TYPING_DELAY)
        pyautogui.press("enter")
        return

    print("Sorry, command not recognised.")


class WhisperAgent:
    """Voice command processor using OpenAI Whisper."""
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the Whisper agent.
        
        Args:
            model_path: Path to the fine-tuned model checkpoint.
        """
        self.model = whisper.load_model("tiny")
        
        if model_path:
            checkpoint = torch.load(model_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
        
        self.sample_rate = SAMPLE_RATE
        self.audio_data = []
        self.recording = False
        self.stream = None

    def callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        """Audio input callback for recording.

        Appends incoming audio data to the audio_data list when recording
        is active.

        Args:
            indata: Input audio data from the microphone.
            frames: Number of frames in the audio buffer.
            time: Timing information for the audio stream.
            status: Status flags for the audio stream.
        """
        if self.recording:
            self.audio_data.append(indata.copy())

    def on_press(self, key: Any) -> None:
        """Handle key press events for starting audio recording.

        Starts audio recording when the CTRL key is pressed and recording is not
        already active. Sets up the audio input stream and begins capturing data.

        Args:
            key: The keyboard key that was pressed.
        """
        try:
            if key == keyboard.Key.ctrl and not self.recording:
                print("Recording... (release CTRL to stop)")
                self.recording = True
                self.audio_data = []
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate, 
                    channels=1, 
                    dtype="float32", 
                    callback=self.callback
                )
                self.stream.start()
        except Exception as e:
            print(f"Error: {e}")

    def on_release(self, key: Any) -> bool:
        """Handle key release events for stopping audio recording.

        Stops audio recording when the CTRL key is released, processes the recorded
        audio through Whisper for transcription, and executes the resulting command.

        Args:
            key: The keyboard key that was released.

        Returns:
            False to end the listener after one recording cycle.
        """
        if key == keyboard.Key.ctrl and self.recording:
            print("Stopped recording.")
            self.recording = False
            self.stream.stop()
            self.stream.close()
            audio = np.concatenate(self.audio_data, axis=0).flatten()
            print("Transcribing...")
            result = self.model.transcribe(audio, fp16=False, language="en")
            print(f"Transcription: {result['text']}")
            execute_command(result["text"])
            return False  # End listener after one recording

    def run(self) -> None:
        """Start the voice command listener."""
        while True:
            print("Hold CTRL to record...")
            with keyboard.Listener(
                on_press=self.on_press, 
                on_release=self.on_release
            ) as listener:
                listener.join()
            print()


def main() -> None:
    """Main entry point for the Whisper Agent."""
    agent = WhisperAgent(model_path=str(CHECKPOINT_PATH))
    agent.run()


if __name__ == "__main__":
    main()
