# Download the datasets and checkpoints
# Requires: pip install gdown

if ! command -v gdown &> /dev/null; then
    echo 'Installing gdown (required for Google Drive downloads)...'
    if ! command -v pipx &> /dev/null; then
        echo 'pipx not found. Install it with: sudo apt install pipx && pipx ensurepath'
        exit 1
    fi
    pipx install gdown
fi

if [ ! -d datasets/ycb/YCB_Video_Dataset ];then
echo 'Downloading the YCB-Video Dataset'
gdown 1if4VoEXNx9W3XCn0Y7Fp15B4GpcYbyYi -O YCB_Video_Dataset.zip \
&& unzip YCB_Video_Dataset.zip \
&& mv YCB_Video_Dataset/ datasets/ycb/ \
&& rm YCB_Video_Dataset.zip
fi

if [ ! -d datasets/linemod/Linemod_preprocessed ];then
echo 'Downloading the preprocessed LineMOD dataset'
gdown 1YFUra533pxS_IHsb9tB87lLoxbcHYXt8 -O Linemod_preprocessed.zip \
&& unzip Linemod_preprocessed.zip \
&& mv Linemod_preprocessed/ datasets/linemod/ \
&& rm Linemod_preprocessed.zip
fi

if [ ! -d trained_checkpoints ];then
echo 'Downloading the trained checkpoints...'
gdown 1bQ9H-fyZplQoNt1qRwdIUX5_3_1pj6US -O trained_checkpoints.zip \
&& unzip trained_checkpoints.zip -x "__MACOSX/*" "*.DS_Store" "*.gitignore" -d trained_checkpoints \
&& mv trained_checkpoints/trained*/ycb trained_checkpoints/ycb \
&& mv trained_checkpoints/trained*/linemod trained_checkpoints/linemod \
&& rm -r trained_checkpoints/trained*/ \
&& rm trained_checkpoints.zip
fi

echo 'done'