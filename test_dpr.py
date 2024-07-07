import os
import csv
import unicodedata
import time
import pickle
import regex as re

import numpy as np 
import pandas as pd
from tqdm import tqdm
import torch
import transformers
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer,BertModel,BertTokenizerFast

from src.utils.tokenizers import SimpleTokenizer
from src.utils import normalize_query

WIKIEPEDIA_PATH = "/workspace/mnt2/dpr_datasets/downloads/data/wikipedia_split/psgs_w100.tsv"
TEST_FILE_DIR="/workspace/mnt2/dpr_datasets/downloads/data/retriever/qas/"
ENCODING_BATCH_SIZE=32
NUM_DOCS=21015324

transformers.logging.set_verbosity_error()

def normalize(text):
    return unicodedata.normalize("NFD", text)

def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answers,doc,is_trec=False):
    if not is_trec:
        tokenizer = SimpleTokenizer()
        doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
        for answer in answers:
            answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
            for i in range(0, len(doc) - len(answer) + 1):
                    if answer == doc[i : i + len(answer)]:
                        return True
    else :
        for answer in answers :
            answer = normalize(answer)
            if regex_match(doc, answer) :
                return True
    return False

if __name__ == '__main__':
    import faiss     
    faiss.omp_set_num_threads(16)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_shards",type=int,default=4)
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--pretrained_model_path",required=True)
    parser.add_argument("--data_split",required=True)
    parser.add_argument("--result_file_path",required=True)
    args = parser.parse_args()

    ## load QA dataset
    query_col,answers_col=0,1
    queries,answers = [],[]
    TEST_FILE = os.path.join(TEST_FILE_DIR,args.data_split+".csv")
    with open(TEST_FILE) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            queries.append(normalize_query(row[query_col]))
            answers.append(eval(row[answers_col]))
    queries = [queries[idx:idx+ENCODING_BATCH_SIZE] for idx in range(0,len(queries),ENCODING_BATCH_SIZE)]
    
    # make faiss index
    embedding_dimension = 768 
    index = faiss.IndexFlatIP(embedding_dimension)
    for idx in tqdm(range(args.num_shards),desc='building index from embedding...'):
        data = np.load(f"{args.embedding_dir}/wikipedia_shard_{idx}.npy")
        index.add(data)  

    ## load wikipedia passages
    id_col,text_col,title_col=0,1,2
    wiki_passages = []
    with open(WIKIEPEDIA_PATH) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in tqdm(reader,total=NUM_DOCS,desc="loading wikipedia passages..."):
            if row[id_col] == "id":continue
            wiki_passages.append(row[text_col].strip('"'))
    
    ## load query encoder
    if args.pretrained_model_path == 'facebook/dpr-question_encoder-single-nq-base':
        query_encoder = DPRQuestionEncoder.from_pretrained(args.pretrained_model_path)
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.pretrained_model_path)
    else:
        query_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()

    ## embed queries
    query_embeddings = []
    for query in tqdm(queries,desc='encoding queries...'):
        with torch.no_grad():
           query_embedding = query_encoder(**tokenizer(query,max_length=256,truncation=True,padding='max_length',return_tensors='pt').to(device))
        if isinstance(query_encoder,DPRQuestionEncoder):
            query_embedding = query_embedding.pooler_output
        else:
            query_embedding = query_embedding.last_hidden_state[:,0,:]
        query_embeddings.append(query_embedding.cpu().detach().numpy())
    query_embeddings = np.concatenate(query_embeddings,axis=0)

    ## retrieve top-k documents
    print("searching index ")
    start_time = time.time()
    top_k = 100
    faiss.omp_set_num_threads(16)
    _,I = index.search(query_embeddings,top_k)
    print(f"takes {time.time()-start_time} s")

    hit_lists = []
    if_trec = "trec" in args.data_split
    for answer_list,id_list in tqdm(zip(answers,I),total=len(answers),desc='calculating metrics...'):
        ## process single query
        hit_list = []
        for doc_id in id_list:
            doc = wiki_passages[doc_id]
            hit_list.append(has_answer(answer_list,doc,if_trec))
        hit_lists.append(hit_list)

    top_k_hits = [0]*top_k
    best_hits = []
    for hit_list in hit_lists:
        best_hit = next((i for i, x in enumerate(hit_list) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    
    top_k_ratio = [x/len(answers) for x in top_k_hits]
    
    test_topk = [4,19,99]

    step_and_epoch = args.pretrained_model_path.split('/')[-2]
    exp_name = args.pretrained_model_path.split('/')[-3]
    epoch = step_and_epoch.split('-')[-1]

    result_dict = {
        "epoch" : epoch,
        "exp_name" : exp_name,
        "top_5" : top_k_ratio[4],
        "top_20" : top_k_ratio[19],
        "top_100" : top_k_ratio[99]
    }

    result_df = pd.DataFrame(result_dict, index=[0])

    if os.path.exists(args.result_file_path) :
        result_df.to_csv(args.result_file_path, mode='a', header=False)
    else :
        result_df.to_csv(args.result_file_path)
    
    print(f"EXP: {exp_name} EPOCH: {epoch}")
    for idx in [4,19,99]:
        print(f"top-{idx+1} accuracy",top_k_ratio[idx])
