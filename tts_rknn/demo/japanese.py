import torch
import os
from TTS.api import TTS

# --- 1. CRITICAL FIX FOR PYTORCH 2.6+ ---
import torch.serialization
orig_load = torch.load
torch.load = lambda *args, **kwargs: orig_load(*args, **{**kwargs, 'weights_only': False})

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Initialize the standard Multilingual XTTS v2
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = TTS(model_name).to(device)

    # 3. Path to a reference voice
    # Since 'Daisy' failed, we provide a real file.
    # You can use the one from your previous Vietnamese download:
    # nu-nhe-nhang.wav	    Female, gentle and soft (the one you used).
    # nam-truyen-cam.wav	Male, inspirational/emotional.
    # nam-calm.wav	        Male, calm and steady.
    # nu-luu-loat.wav	    Female, fluent and professional.
    # nam-nhanh.wav	        Male, fast-paced (good for news).
    # nu-nhan-nha.wav	    Female, slow and deliberate.        
    speaker_wav = "models/viXTTS/samples/nu-nhe-nhang.wav"
    
    if not os.path.exists(speaker_wav):
        print(f"❌ Error: Please provide a valid path to a .wav file in 'speaker_wav'")
        return

    # 4. Japanese Text
    text = "大阪メトロは２０日、２０２５年に大阪万博が開催される人工島「夢洲（ゆめしま）」（大阪市此花区）に開業する新駅の構想を発表した。高さ２５０メートル超のタワービルが一体となった施設で、総工費は１千億円を超える見込みだという。カジノを含む統合型リゾート（ＩＲ）の誘致を前提として、２４年度中の開業をめざす。"
    output_path = "japanese_fixed.wav"

    print(f"> Generating Japanese audio using: {speaker_wav}")
    
    # Use 'speaker_wav' instead of 'speaker'
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="ja",
        file_path=output_path
    )

    print(f"> Success! Audio saved to {output_path}")

if __name__ == "__main__":
    main()