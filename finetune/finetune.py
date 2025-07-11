"""Fine-tuning script for Whisper model on voice command data.

This script fine-tunes a Whisper model to better recognize voice commands
for Google searches and ChatGPT interactions.
"""

import os
import re
import string
from pathlib import Path

import torch
import whisper
from tqdm import tqdm

# Constants
RECORDINGS_DIR = Path("voice_recordings")
CHECKPOINT_DIR = Path("checkpoints")
MODEL_NAME = "tiny"
EPOCHS = 2
LEARNING_RATE = 1e-5
OUTPUT_MODEL_NAME = "whisper_tiny_finetuned_new.pt"
LOG_FILE_NAME = "fine_tune_logs.txt"

# Audio file configuration
AUDIO_COUNT = 10  # Change to the number of audio files you have recorded for fine-tuning (10+ recommended)
AUDIO_FORMAT = ".m4a"  # Change to your file format if different


def clean_for_compare(text):
    """Clean and normalize text for comparison.

    Converts text to lowercase, normalizes 'chatgpt' variations to 'chat gpt',
    removes punctuation, and normalizes whitespace to enable accurate comparison
    between predicted and ground truth transcriptions.

    Args:
        text (str): The input text to clean and normalize.

    Returns:
        str: The cleaned and normalized text.
    """
    text = text.lower().strip()

    # Replace 'chatgpt' with 'chat gpt' (handles chat-gpt too)
    text = re.sub(r"chat[-\s]?gpt", "chat gpt", text, flags=re.IGNORECASE)

    # Remove punctuation except for word merges (so "chatgpt" can become "chat gpt")
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)

    text = re.sub(r"chatgpt", "chat gpt", text)

    # Normalize all whitespace to single spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_audio_files():
    """Get list of audio files to process.
    
    Returns:
        list: List of absolute paths to audio files.
    """
    audio_files = [
        f"audio_{i:02d}{AUDIO_FORMAT}" for i in range(1, AUDIO_COUNT + 1)
    ]
    return [RECORDINGS_DIR / fname for fname in audio_files]


def get_ground_truths():
    """Get ground truth transcriptions for audio files.
    
    Returns:
        list: List of ground truth transcriptions.
    """
    return [
        "Ask chat GPT to tell us a joke",
        "Search Google for pizza places near me",
        "Ask chat GPT to write me a poem",
        "Search Google how to bake a cake",
        "Ask chat GPT what the best coding language is",
        "Search Google for what's the meaning of life",
        "Ask chat GPT to tell me a really funny joke",
        "Search Google for indian restaurants near me",
        "Ask chat GPT to tell me a joke",
        "Search Google for greek restaurants near me",
    ]


def find_mismatches(model, audio_files, ground_truths):
    """Find mismatches between Whisper predictions and ground truth.
    
    Args:
        model: Whisper model instance.
        audio_files (list): List of audio file paths.
        ground_truths (list): List of ground truth transcriptions.
        
    Returns:
        list: List of mismatch dictionaries.
    """
    mismatches = []

    for idx, (audio_path, target_text) in tqdm(
        enumerate(zip(audio_files, ground_truths)),
        total=len(audio_files),
        desc="Checking transcriptions"
    ):
        result = model.transcribe(str(audio_path), language="en", fp16=False)
        pred_text = result["text"]
        gt_text = target_text

        # Clean for comparison
        pred_cmp = clean_for_compare(pred_text)
        gt_cmp = clean_for_compare(gt_text)

        if pred_cmp != gt_cmp:
            tqdm.write(f"Mismatch at {audio_path}:")
            tqdm.write(f"  Whisper : {pred_text!r}")
            tqdm.write(f"  Ground  : {gt_text!r}")

            mismatches.append({
                "audio_path": str(audio_path),
                "target_text": target_text,
                "original_whisper": pred_text,
            })

    return mismatches


def train_model(model, tokenizer, mismatches):
    """Train the model on mismatched samples.
    
    Args:
        model: Whisper model instance.
        tokenizer: Whisper tokenizer instance.
        mismatches (list): List of mismatch dictionaries.
        
    Returns:
        tuple: (losses, outputs, optimizer) from training.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    outputs = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for m in tqdm(mismatches, desc="Fine-tuning (only mismatches)"):
            audio_path = m["audio_path"]
            target_text = m["target_text"]
            original_whisper = m["original_whisper"]

            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # Build token sequence
            ids = [
                tokenizer.sot,
                tokenizer.language_token,
                tokenizer.transcribe,
                tokenizer.no_timestamps,
            ]
            ids += tokenizer.encode(target_text)
            ids += [tokenizer.eot]

            model.train()
            tokens = torch.tensor(ids).unsqueeze(0).to(model.device)
            mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)
            pred = model(tokens=tokens, mel=mel)
            target = tokens[:, 1:].contiguous()
            pred = pred[:, :-1, :].contiguous()

            sample_loss = loss_fn(pred.transpose(1, 2), target).item()
            losses.append(sample_loss)
            
            outputs.append({
                "idx": ids,
                "loss": sample_loss,
                "trgt_ids": target.squeeze().tolist(),
                "pred_ids": torch.argmax(pred, dim=2).squeeze().tolist(),
                "trgt_txt": tokenizer.decode(target.squeeze().tolist()),
                "original_whisper": original_whisper,
                "pred_txt": tokenizer.decode(torch.argmax(pred, dim=2).squeeze().tolist()),
                "audio_path": audio_path,
                "target_text": target_text,
            })

            loss = loss_fn(pred.transpose(1, 2), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses, outputs, optimizer


def save_model_and_logs(model, optimizer, outputs):
    """Save the trained model and training logs.
    
    Args:
        model: Trained Whisper model.
        optimizer: Model optimizer.
        outputs (list): Training outputs and logs.
    """
    # Ensure checkpoint directory exists
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    # Save model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        CHECKPOINT_DIR / OUTPUT_MODEL_NAME,
    )

    # Save logs
    with open(LOG_FILE_NAME, "w") as f:
        for o in outputs:
            f.write(f"{os.path.basename(o['audio_path'])}\n")
            f.write(f"Target IDs: {o['trgt_ids']}\n")
            f.write(f"Predicted IDs: {o['pred_ids']}\n")
            f.write(f"Target TXT: {o['trgt_txt']}\n")
            f.write(f"Original Whisper: {o['original_whisper']}\n")
            f.write(f"Predicted TXT: {o['pred_txt']}\n")
            f.write("-" * 40 + "\n")


def main():
    """Main fine-tuning function."""
    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME)
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)

    print("Getting audio files and ground truths...")
    audio_files = get_audio_files()
    ground_truths = get_ground_truths()

    print("Finding mismatches...")
    mismatches = find_mismatches(model, audio_files, ground_truths)
    print(f"Found {len(mismatches)} mismatches.")

    if not mismatches:
        print("No mismatches found. Fine-tuning not needed.")
        return

    print("Starting fine-tuning...")
    losses, outputs, optimizer = train_model(model, tokenizer, mismatches)

    print("Saving model and logs...")
    save_model_and_logs(model, optimizer, outputs)

    print(f"\nFine-tuning complete. {len(outputs)} mismatched samples were trained.")
    print(f"Model saved as '{OUTPUT_MODEL_NAME}' in {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
