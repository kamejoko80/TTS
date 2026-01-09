import torch
import os
from TTS.api import TTS

# 
# To run vietnamese demo:
#     pip install huggingface_hub
#     python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='capleaf/viXTTS', local_dir='models/viXTTS')" 
#     python3 vietnamese.py
#

# --- 1. CRITICAL FIX FOR PYTORCH 2.6+ ---
import torch.serialization
orig_load = torch.load
torch.load = lambda *args, **kwargs: orig_load(*args, **{**kwargs, 'weights_only': False})

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path to your downloaded Vietnamese model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_path = os.path.join(base_dir, "models/viXTTS")
    config_path = os.path.join(local_model_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"❌ ERROR: Model not found at {local_model_path}")
        return

    print("> Loading Vietnamese XTTS model...")
    # Initialize TTS
    tts = TTS(model_path=local_model_path, config_path=config_path).to(device)

    # --- 2. DYNAMIC VIETNAMESE SUPPORT FIX ---
    # Instead of importing the Tokenizer, we modify the one already inside the model
    def patched_preprocess_text(txt, lang):
        # When lang is 'vi', we just return the raw text to bypass the 'NotImplemented' error
        return txt

    # Inject the patch into the loaded model's tokenizer
    if hasattr(tts.synthesizer.tts_model, "tokenizer"):
        tts.synthesizer.tts_model.tokenizer.preprocess_text = patched_preprocess_text
        print("> Vietnamese language check bypassed successfully.")

    # 3. Find a reference voice
    # nu-nhe-nhang.wav	    Female, gentle and soft (the one you used).
    # nam-truyen-cam.wav	Male, inspirational/emotional.
    # nam-calm.wav	        Male, calm and steady.
    # nu-luu-loat.wav	    Female, fluent and professional.
    # nam-nhanh.wav	        Male, fast-paced (good for news).
    # nu-nhan-nha.wav	    Female, slow and deliberate.
        
    speaker_wav = os.path.join(local_model_path, "samples/nam-truyen-cam.wav")
    if not os.path.exists(speaker_wav):
        # Fallback to find any wav in the folder
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                if file.endswith(".wav"):
                    speaker_wav = os.path.join(root, file)
                    break

    # 4. Run TTS in Vietnamese
    output_path = "tieng_viet_test.wav"
    # Using a longer sentence for better quality
    text = "Dù chỉ có môt mình đi kiện ông trời những Cóc tía không hề nan lòng. Đi qua một vũng đầm khô, Cóc tía gặp Cua càng. Cua hỏi Cóc đi đâu. Cóc bèn kể rõ sự tình và rủ Cua cùng đi kiện Trời. Ban đầu Cua định bàn ngang, thà chết ở đây còn hơn chứ Trời xa thế đi sao tới mà kiện với tụng. Nhưng những con vật ở quanh Cua nghe Cóc nói lại tranh nhau mà bàn ngang bàn lùi, làm cho Cua nổi giận. Nói ngang bàn ngang là chuyện ngang của Cua thế mà họ lại dám tranh mất cái quyền ấy, cái quyền được phép ngang như cua cơ mà. Thế là Cua làm ngược lại, Cua tình nguyện cùng đi với Cóc."

    print(f"> Đang tạo âm thanh với file mẫu: {os.path.basename(speaker_wav)}")
    
    try:
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language="vi",
            file_path=output_path
        )
        print(f"> Thành công! File đã lưu tại: {output_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tạo âm thanh: {e}")

if __name__ == "__main__":
    main()