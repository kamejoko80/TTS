import torch
from TTS.api import TTS

# --- CRITICAL FIX FOR PYTORCH 2.6+ ---
import torch.serialization
orig_load = torch.load
torch.load = lambda *args, **kwargs: orig_load(*args, **{**kwargs, 'weights_only': False})
# -------------------------------------

def main():
    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"> Using device: {device}")

    # Initialize a high-quality single-speaker model (Jenny)
    # This model does NOT require a reference wav file.
    model_name = "tts_models/en/jenny/jenny"
    tts = TTS(model_name).to(device)

    output_path = "test_jenny.wav"
    text = "This is a test using the Jenny model. It works instantly because it does not require a voice sample."

    print("> Generating audio...")
    tts.tts_to_file(text=text, file_path=output_path)

    print(f"> Success! Audio saved to {output_path}")

if __name__ == "__main__":
    main()