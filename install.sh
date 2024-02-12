mkdir -p app
cd src
python -m pip install --upgrade pip
python -m pip install -r ./requirements.txt
python -m pip install -q torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 torchtext==0.16.0 torchdata==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121 -U
python -m pip install imageio-ffmpeg numpy==1.23.0 pandas pyngrok deepspeed==0.12.6 decord==0.6.0 transformers==4.37.0 einops timm tiktoken accelerate mpi4py
cd ../app
FOLDER_PATH=MoE-LLaVA-hf
if ! [ -d "$FOLDER_PATH" ]; then
    git clone -b dev https://github.com/camenduru/MoE-LLaVA-hf
    cd $FOLDER_PATH
else
    cd $FOLDER_PATH
    git pull
fi
python -m pip install -e .
cd ..
python -m python download_model.py