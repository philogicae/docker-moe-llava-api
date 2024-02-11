FOLDER_PATH=./MoE-LLaVA
if ! [ -d "$FOLDER_PATH" ]; then
    git clone https://github.com/PKU-YuanGroup/MoE-LLaVA
    cd $FOLDER_PATH
else
    cd $FOLDER_PATH
    git pull
fi
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install flash-attn --no-build-isolation
cd ..
python -m pip install -r requirements.txt