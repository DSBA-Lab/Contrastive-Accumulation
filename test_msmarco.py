import os
import json 
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import transformers
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset

from src.msmarco_utils.search import FlatIPFaissSearch, CustomBiEncoder

def convert_qrels_beir(qrels):
    new_qrels = defaultdict(dict)
    for qrel in qrels:
        if qrel['score'] > 0:
            new_qrels[str(qrel['query-id'])][str(qrel['corpus-id'])] = qrel['score']
    return new_qrels

transformers.logging.set_verbosity_error()
NUM_SHARDS = 4

if __name__ == '__main__':
    import faiss     
    faiss.omp_set_num_threads(16)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--model_save_dir",required=True)
    parser.add_argument("--data_split",required=True, choices=["validation", "test"])
    parser.add_argument("--result_file_path",required=True)
    args = parser.parse_args()
    print(f"""
    Arguments:
    embedding_dir: {args.embedding_dir}
    model_save_dir: {args.model_save_dir}
    data_split: {args.data_split}
    result_file_path: {args.result_file_path}
          """)
    faiss.omp_set_num_threads(16)


    ## load QA dataset
    with open(f'/workspace/mnt2/dpr_datasets/msmarco/preprocessed/msmarco_{args.data_split}.json', 'r') as f:
        queries = json.load(f)
    queries = {query['question']['_id'] : query['question']['text'] for query in queries}
    
    corpus = load_dataset('BeIR/msmarco', 'corpus', cache_dir='/workspace/mnt2/dpr_datasets/msmarco/original')["corpus"]
    qrels = load_dataset('BeIR/msmarco-qrels', cache_dir='/workspace/mnt2/dpr_datasets/msmarco/original')[args.data_split]
    qrels = convert_qrels_beir(qrels)

    # make faiss index
    model = CustomBiEncoder(model_save_dir=args.model_save_dir)
    index_model = FlatIPFaissSearch(model, batch_size=1024, output_dir=args.embedding_dir)
    # from here
    index_model.load_and_index(embed_dir=args.embedding_dir, mapping_dict_dir=os.path.join(args.embedding_dir, 'mapping_dic.tsv'))
    retriever = EvaluateRetrieval(index_model, score_function="dot")
    results = retriever.retrieve(corpus, queries, )

    k_values = [5, 10, 20, 100, 1000]
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    print(f"ndcg: {ndcg}, map: {_map}, recall: {recall}, precision: {precision}, mrr: {mrr}")

  
    index_name = "/".join(args.embedding_dir.split('/')[-2:])
    print(f"""
    Results saved at {args.result_file_path}
    Row Name : {index_name}
          """)
    result_df = pd.DataFrame(dict(**ndcg, **_map, **recall, **precision, **mrr), index=[index_name])
    if os.path.exists(args.result_file_path):
        result_df.to_csv(args.result_file_path, mode='a', header=False)
    else:
        result_df.to_csv(args.result_file_path)