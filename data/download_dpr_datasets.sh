# wikipedia corpus
python src/utils/download_data.py --resource data.wikipedia_split.psgs_w100 --output_dir /workspace/mnt2/dpr_datasets

# natural questions
python src/utils/download_data.py --resource data.retriever.nq-train --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.nq-dev --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.qas.nq-dev --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.qas.nq-test --output_dir /workspace/mnt2/dpr_datasets

# trivia
python src/utils/download_data.py --resource data.retriever.trivia-train --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.qas.trivia-test --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.qas.trivia-dev --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.trivia-dev --output_dir /workspace/mnt2/dpr_datasets

# web questions
python src/utils/download_data.py --resource data.retriever.webq-train --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.qas.webq-test --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.webq-dev --output_dir /workspace/mnt2/dpr_datasets

# trec
python src/utils/download_data.py --resource data.retriever.curatedtrec-train --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.qas.curatedtrec-test --output_dir /workspace/mnt2/dpr_datasets
python src/utils/download_data.py --resource data.retriever.curatedtrec-dev --output_dir /workspace/mnt2/dpr_datasets
