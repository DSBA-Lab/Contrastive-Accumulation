import csv
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizerFast,
    BertModel,
    )
import torch
import numpy as np
from accelerate import PartialState

if __name__ == "__main__":
    NUM_DOCS = 21015324
    WIKIPEDIA_PATH = "/workspace/mnt2/dpr_datasets/downloads/data/wikipedia_split/psgs_w100.tsv"
    ENCODING_BATCH_SIZE = 1024

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir",type=str)
    parser.add_argument("--model_save_dir",required=True)
    parser.add_argument("--log_dir",type=str)
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    ## load encoder
    if args.model_save_dir == 'facebook/dpr-ctx_encoder-single-nq-base':
        doc_encoder = DPRContextEncoder.from_pretrained(args.model_save_dir)
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.model_save_dir)
    else:
        doc_encoder = BertModel.from_pretrained(args.model_save_dir,add_pooling_layer=False)
        tokenizer = BertTokenizerFast.from_pretrained(args.model_save_dir)
    doc_encoder.eval()
    doc_encoder.to(device)


    ## load wikipedia passages
    progress_bar = tqdm(total=NUM_DOCS, disable=not distributed_state.is_main_process,ncols=100,desc='loading wikipedia...')
    id_col,text_col,title_col=0,1,2
    wikipedia = []
    with open(WIKIPEDIA_PATH) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[id_col] == "id":continue
            wikipedia.append(
                [row[title_col],row[text_col].strip('"')]
            )
            progress_bar.update(1)

    with distributed_state.split_between_processes(wikipedia) as sharded_wikipedia:
        
        sharded_wikipedia = [sharded_wikipedia[idx:idx+ENCODING_BATCH_SIZE] for idx in range(0,len(sharded_wikipedia),ENCODING_BATCH_SIZE)]
        encoding_progress_bar = tqdm(total=len(sharded_wikipedia), disable=not distributed_state.is_main_process,ncols=100,desc='encoding wikipedia...')
        doc_embeddings = []
        for data in sharded_wikipedia:
            title = [x[0] for x in data]
            passage = [x[1] for x in data]
            model_input = tokenizer(title,passage,max_length=256,padding='max_length',return_tensors='pt',truncation=True).to(device)
            with torch.no_grad():
                if isinstance(doc_encoder,BertModel):
                    CLS_POS = 0
                    output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
                else:
                    output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(args.embed_dir,exist_ok=True)
        np.save(f'{args.embed_dir}/wikipedia_shard_{distributed_state.process_index}.npy',doc_embeddings)