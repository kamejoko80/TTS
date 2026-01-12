
## 🐸Coqui.ai running on RK3588 platform build guideline:

Execution flow:

    Text → mel (model_x) (CPU) → waveform (HiFiGAN) (NPU)

## Build guideline (on Linux x86 desktop PC):

```bash
mkdir Coqui_TTS
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip
pip install torchcodec mecab-python3 unidic-lite
```

```bash
git clone git@github.com:kamejoko80/TTS.git
cd TTS
git checkout henry_rk3588
make system-deps
make install
```

```bash
cd tts_rknn/demo
pip install huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='capleaf/viXTTS', local_dir='models/viXTTS')"
python3 english.py
python3 japanese.py
python3 vietnamese.py
```

When running if there is an error like: ImportError: cannot import name 'BeamSearchScorer' from 'transformers' then run the bellow commands to fix:

```bash
pip uninstall transformers -y
pip install transformers==4.36.2
```

## Export ONNX model:

```bash
pip install onnxscript onnxruntime
pip uninstall numpy
pip install "numpy<2.0"
python3 export_hifigan_v2_onnx.py --mel_frames 1200 --out models/hifigan_v2_T1200.onnx
```

Run the bellow script to verify the onnx model:


```bash
python3 tts_cpu_mel_onnx_hifigan_x86.py --vocoder_onnx models/hifigan_v2_T1200.onnx --tts_model tts_models/en/ljspeech/tacotron2-DDC --text "Did you ever hear a folk tale about a giant turtle?" --mel_frames 1200  --out_wav test_onnx_hifigan.wav
python3 tts_cpu_mel_onnx_hifigan_x86.py --vocoder_onnx models/hifigan_v2_T1200.onnx --tts_model tts_models/en/ljspeech/fast_pitch --text "Did you ever hear a folk tale about a giant turtle?" --mel_frames 1200  --out_wav test_onnx_hifigan.wav
```

## Export RKNN model:

Must open a different linux terminal to install the RKNN-Toolkit2 on the Linux x86 desktop PC

```bash
cd Coqui_TTS
mkdir RKNN-Toolkit2
cp TTS/tts_rknn/sh/Miniforge3-Linux-x86_64.sh ./RKNN-Toolkit2/Miniforge3-Linux-x86_64.sh
cd RKNN-Toolkit2
```

Run bash Miniforge3-Linux-x86_64.sh (availabe in the repo's scripts folder) and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.8 version:

```bash
conda create -n RKNN-Toolkit2 python=3.8
```

Activate RKNN-Toolkit2:

```bash
conda activate RKNN-Toolkit2
```

To deactivate:

```bash
conda deactivate
```

Install RKNN-Toolkit2 from github repo:

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2
pip install -r rknn-toolkit2/packages/x86_64/requirements_cp38-2.3.2.txt
pip install rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Run the below command to export the RKNN model

```bash
cd Coqui_TTS/TTS/tts_rknn
python3 convert_hifigan_v2_onnx_rknn.py --onnx models/hifigan_v2_T1200.onnx --out models/hifigan_v2_T1200.rknn --platform rk3588 --n_mels 80 --mel_frames 1200
```


## Setup on RK3588:

```bash
mkdir Coqui_TTS
git clone https://github.com/kamejoko80/TTS.git
cd TTS
git checkout henry_rk3588
cd ..
cp TTS/tts_rknn/sh/Miniforge3-25.11.0-0-Linux-aarch64.sh ./
```

Run bash Miniforge3-25.11.0-0-Linux-aarch64.sh and install in path = $PWD/env

Every time we open a new console we must activate the env:

```bash
source env/bin/activate
```

Create a Conda environment named "RKNN-Toolkit2" with Python 3.10 version:

```bash
conda create -n RKNN-Toolkit2 python=3.10
```

Activate RKNN-Toolkit2:

```bash
> conda activate RKNN-Toolkit2
```

To deactivate:

```bash
> conda deactivate
```

Install RKNN-Toolkit2 & Coqui_TTS:

```bash
pip install rknn-toolkit-lite2
```

Install rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

Install Coqui_TTS:

```bash
cd Coqui_TTS/TTS
make system-deps
make install
```

Test torch inference:

```bash
cd Coqui_TTS/TTS/tts_rknn/demo
python3 english.py
```

Test MeloTTS with RKNN accelerator:

Copy "hifigan_v2_T1200.rknn" from the Linux x86 PC into the Coqui_TTS/TTS/tts_rknn/models folder, then run:

```bash
cd Coqui_TTS/TTS/tts_rknn
python3 tts_cpu_mel_rknn_hifigan_rk3588.py --vocoder_rknn models/hifigan_v2_T1200.rknn --tts_model tts_models/en/ljspeech/tacotron2-DDC --out_wav rknn_tts_tacotron2.wav --mel_frames 1200 --speed 1.0 --tail_ms 30 --text "This sentence is synthesized entirely on the RK3588 NPU."
python3 tts_cpu_mel_rknn_hifigan_rk3588.py --vocoder_rknn models/hifigan_v2_T1200.rknn --tts_model tts_models/en/ljspeech/fast_pitch --out_wav rknn_tts_fast_pitch.wav --mel_frames 1200 --speed 1.0 --tail_ms 30 --text "This sentence is synthesized entirely on the RK3588 NPU."
python3 tts_cpu_mel_rknn_hifigan_text_chunking_rk3588.py --vocoder_rknn models/hifigan_v2_T1200.rknn --tts_model tts_models/en/ljspeech/fast_pitch --out_wav rknn_tts_fast_pitch.wav --mel_frames 1200 --speed 1.0 --tail_ms 30 \ 
--text "AI technology enables machines to perform tasks needing human intelligence, like learning, reasoning, problem-solving, and understanding language, by processing data to recognize patterns, make decisions, and adapt, powering everything from virtual assistants and recommendation systems to self-driving cars and medical diagnostics. It's a field of computer science focused on creating smart systems that learn from data rather than explicit programming, improving performance over time. "
```