

from src.msmarco_utils.search import FlatIPFaissSearch, CustomBiEncoder

from datasets import load_dataset

if __name__ == "__main__" :
    from tqdm import tqdm
    import argparse
    import os

    MSMARCO_CORPUS = load_dataset('BeIR/msmarco', 'corpus', cache_dir='/workspace/mnt2/dpr_datasets/msmarco/original')["corpus"]
    print("MSMARCO_CORPUS loading")
    MSMARCO_CORPUS = {sample['_id'] : {"title" : sample['title'], "text" : sample['text']} for sample in tqdm(MSMARCO_CORPUS)}
    BATCH_SIZE=1024

    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir",type=str)
    parser.add_argument("--model_save_dir",required=True)
    args = parser.parse_args()

    print(f"""
          Arguments:
            embed_dir: {args.embed_dir}
            model_save_dir: {args.model_save_dir}
            """
          )

    os.makedirs(args.embed_dir, exist_ok=True)

    model = CustomBiEncoder(model_save_dir=args.model_save_dir)
    index_model = FlatIPFaissSearch(model, batch_size=BATCH_SIZE, output_dir=args.embed_dir)
    index_model.embed_and_save(MSMARCO_CORPUS, score_function='dot')