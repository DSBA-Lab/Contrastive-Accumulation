DEVICE=$1
SAVED_MODEL_PATH=$2
EMBED_DIR=$3
DATA_SPLIT=$4
RESULT_FILE_PATH=$5

export CUDA_VISIBLE_DEVICES=$DEVICE
python test_dpr.py \
    --embedding_dir $EMBED_DIR/embeddings \
    --pretrained_model_path $SAVED_MODEL_PATH/query_encoder \
    --data_split $DATA_SPLIT \
    --result_file_path $RESULT_FILE_PATH
