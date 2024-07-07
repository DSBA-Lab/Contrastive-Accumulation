PRETRAINED_MODEL_PATH=$1
OUTPUT_DIR=$2
accelerate launch --num_processes=1 doc2embedding.py \
    --model_save_dir $PRETRAINED_MODEL_PATH/doc_encoder \
    --embed_dir $OUTPUT_DIR/embeddings