# 🤖 Mini Whisper Agent

> **A fun 24-hour project:** A voice-controlled assistant that uses OpenAI Whisper (tiny) for speech recognition to execute voice commands like Google searches and ChatGPT interactions.

## Video Demo

https://github.com/user-attachments/assets/35188d99-1010-46cd-9b02-1b8ed182f84f

## Features

- **Voice Command Recognition**: Uses fine-tuned Whisper tiny model for accurate speech-to-text
- **Google Search**: Voice commands like "Search Google for pizza places near me" 
- **ChatGPT Integration**: Voice commands like "Ask ChatGPT to tell me a joke"
- **Real-time Processing**: Hold CTRL to record, release to process and execute commands
- **Browser Compatibility**: Works great with Brave browser (Chrome may encounter verification issues with ChatGPT)

## Supported Voice Commands

The system currently recognises these command patterns:

**Google Search Commands:**
- "Search Google for [your query]"
- "Google for [your query]" 
- "Search for [your query]"

**ChatGPT Commands:**
- "Ask ChatGPT to [your prompt]"
- "Ask ChatGPT [your prompt]"

*Want more commands? Add them at your leisure! The code is designed for easy extension.*

## Project Structure

```
mini-whisper-agent/
├── app/
│   └── app.py                    # Main voice assistant application
├── finetune/
│   └── finetune.py              # Model fine-tuning script
├── checkpoints/
│   └── whisper_tiny_finetuned.pt # Pre-trained model (via Git LFS)
├── voice_recordings/            # Your training audio files
│   └── tmp/                     # Temporary audio processing
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT license
├── README.md                   # This file
├── pyproject.toml              # Project configuration
├── requirements.txt            # Python dependencies
└── uv.lock                     # UV dependency lock file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ajamesl/mini-whisper-agent.git
cd mini-whisper-agent
git lfs pull  # Download the pre-trained model
```

2. Install system dependencies:
```bash
# Install ffmpeg (required for audio processing)
sudo apt install ffmpeg
```

3. Install Python dependencie (uv recommended):
```bash
uv sync
# OR if using pip:
pip install -r requirements.txt
```

## Usage

### Running the Voice Assistant

🎯 **Want to try it immediately?** The repository includes a pre-trained model ready to use!

```bash
# Run from project root
uv run app/app.py  # or python app/app.py
```

- Hold **CTRL** to start recording your voice command
- Release **CTRL** to stop recording and process the command
- The system will automatically open your browser and execute the command if transcribed correctly

**Browser Compatibility Notes:**
- ✅ **Brave Browser**: Recommended - works seamlessly with all features
- ⚠️ **Chrome**: May encounter issues (e.g., verification issues when accessing ChatGPT)
- 🔧 **Other Browsers**: Not tested

### Fine-tuning the Model

🎤 **For best results, record your own audio clips!** This personalizes the model to your voice and environment.

1. **Record Training Data**: 
   - Use your device microphone to record phrases like "Search Google for..." and "Ask ChatGPT to..."
   - Record in different environments (quiet room, with background noise, etc.) for robustness
   - Save files as `audio_01.m4a`, `audio_02.m4a`, etc. in the `voice_recordings/` folder
   - Aim for 10+ recordings for best fine-tuning results

2. **Update Ground Truth**: Edit the `get_ground_truths()` function in `finetune/finetune.py` to match your recorded phrases

3. **Run Fine-tuning**:
```bash
# Run from project root
uv run finetune/finetune.py  # or python finetune/finetune.py
```

The script will:
- Compare your recordings with the base Whisper model
- Identify mismatches that need correction
- Fine-tune only on the mismatched samples
- Save the improved model to `checkpoints/whisper_tiny_finetuned.pt`

> 💡 **Tip**: The more diverse your training audio (different background noise, speaking speeds, etc.), the more robust your model will be!

## Configuration

### Constants in `app/app.py`:
- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `CHECKPOINT_PATH`: Path to fine-tuned model
- `BROWSER_LOAD_DELAY`: Time to wait for browser to load
- `TYPING_DELAY`: Delay between typing characters

### Constants in `finetune/finetune.py`:
- `MODEL_NAME`: Base Whisper model to fine-tune (default: "tiny")
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Training learning rate
- `AUDIO_COUNT`: Number of audio files to process
- `AUDIO_FORMAT`: Audio file format (e.g., ".m4a", ".wav")

### Running Scripts

Both scripts can be run from the project root directory:

```bash
# Voice assistant
uv run app/app.py

# Fine-tuning
uv run finetune/finetune.py
```

## Extending the System

Want to add more voice commands? Here's how:

1. **Add New Patterns**: Update the regex patterns in `app.py` for your new commands
2. **Implement Handlers**: Add corresponding action functions for your commands  
3. **Record Training Data**: Create audio samples for your new commands
4. **Fine-tune**: Re-run the fine-tuning process with your expanded dataset

The modular design makes it easy to add features like:
- Play/pause music
- Volume control
- Email sending
- Calendar management
- And much more!

## Voice Commands

The system recognises these command patterns:

**Google Search Commands:**
- "Search Google for [query]"
- "Google for [query]" 
- "Search for [query]"

**ChatGPT Commands:**
- "Ask ChatGPT to [prompt]"
- "Ask ChatGPT [prompt]"

*🎯 Pro tip: Speak clearly and naturally - the fine-tuned model adapts to your speaking style!*


### Architecture

- **WhisperAgent Class**: Encapsulates audio recording and speech recognition
- **Command Execution**: Pattern matching for different voice commands
- **Fine-tuning Pipeline**: Modular functions for model training
- **Constants**: Centralized configuration management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Arjuna James

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
