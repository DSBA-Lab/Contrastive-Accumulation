# A Gradient Accumulation Method for Dense Retriever under Memory Constraint

This repository is the official implementation of [A Gradient Accumulation Method for Dense Retriever under Memory Constraint](https://arxiv.org/abs/2406.12356). It is adapted from the repository [nanoDPR](https://github.com/Hannibal046/nanoDPR/tree/master), which offers a simplified replication of the DPR model. 

## 1. Requirements
---
To install the required packages:
```setup
pip install -r requirements.txt
```

## 2. Preparing the data

### 2-1. Download DPR data
---
DPR provides preprocessed datasets in their official repository. Download the datasets with the following command:
```bash
bash data/download_dpr_datasets.sh
```

### 2-2. Download and preprocess MS Marco data
---
You can download and preprocess the MS Marco data using the provided scripts. The BEIR repository and Huggingface offer preprocessed MS Marco data. Additionally, you can filter hard negatives by cross-encoder scores.

Find the download and preprocessing code in `data/msmarco_download_and_preprocess.ipynb`.

## 3. Training
---
You can train the DPR model under various settings:
### 3-1. DPR with ContAccum in low-resource
```bash
python src/train_dpr.py --config_file config/{data_name}/train_dpr_{data_name}_contAccum_cache1_accum4.yaml
```

### 3-2. DPR in high-resource
```bash
python src/train_dpr.py --config_file config/{data_name}/train_dpr_{data_name}_bsz128.yaml
```

### 3-3. DPR in low-resource
```bash
python src/train_dpr.py --config_file config/{data_name}/vram11/train_dpr_{data_name}_bsz8.yaml
```

### 3-4. DPR with gradient accumulation in low-resource
```bash
python src/train_dpr.py --config_file config/{data_name}/vram11/train_dpr_{data_name}_gradAccum_4.yaml
```

## 4. 4. Extracting Embeddings of all passages
### For MS Marco
```bash
accelerate launch --num_processes=4 doc2embedding_msmarco.py \
    --embed_dir /workspace/mnt2/dpr_output/{embed_dir} \
    --model_save_dir /workspace/mnt2/dpr_logs/{model_dir}
```
### For DPR datasets
```bash
bash scripts/tools/embed.sh {model_dir} {embed_dir} 
```

## 5. Evaluation
### For MS Marco
```bash
python test_msmarco.py \
    --embedding_dir {embed_dir} \
    --model_save_dir {model_dir} \
    --data_split test \
    --result_file_path result.csv
```

### For DPR datasets
```bash
bash scripts/tools/test.sh 6 {model_dir}/query_encoder {embed_dir}/embeddings
```

## 6. Implementation details
### 6-1. Traditaional InfoNCE Loss
```python
# q_local: query representations in the same batch 
# p_local: passage representations in the same batch
# labels: n x n matrix that has diagonal 1-hot element
for batch in dataloader:
	q_local, p_local = model(batch)
    sim_matrix = torch.matmul(q_local, p_local.permute(1,0))
	labels = torch.cat([torch.arange(single_device_query_num) + gpu_index * single_device_doc_num for gpu_index in range(accelerator.num_processes)],dim=0).to(matching_score.device)
	loss = F.nll_loss(input=F.log_softmax(sim_matrix,dim=1),target=labels)
    loss.backward()
    ...
```

### 6-2. ContAccum Implementation
```python
# q_local: query representations in the same batch 
# p_local: passage representations in the same batch
# labels: n x n matrix with diagonal 1-hot elements
loss_calculator = LossCalculator(args,hard_neg=args.use_hard_neg)
for batch in dataloader:
	q_local, p_local = model(batch)    
	loss = loss_calculator(q_local, p_local)
    loss.backward()
    if step % gradient_accumulations_step == 0:
    	optimizer.step()
        optimizer.zero_grad()
        ...
```
### 6-3. Hyperparameters for ContAccum
All hyperparameters for ContAccum are contained in the `args` variable:

	•	prev_cache (boolean): Whether to cache the representations generated by the previous model. If not, the memory bank is cleared out after every model update.
	•	cache_query (boolean): Whether to cache the query representations. If not, only the passage representations are cached.
	•	cache_hard_neg (boolean): Whether to cache the hard negative passage representations. This makes the size of the passage memory bank twice as large as the query memory bank.
	•	cache_size (int): The memory bank size. It should be the same as the local batch size.
	•	use_hard_neg (boolean): Whether hard negatives are used for training. This is different from the cache_hard_neg parameter.
	