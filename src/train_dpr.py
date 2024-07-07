## built-in
import math,logging,functools,os
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
import transformers
from transformers import (
    BertTokenizer,
    BertModel,
)
transformers.logging.set_verbosity_error()
logging.basicConfig(
            format      = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt     = '%m/%d/%Y %H:%M:%S',
            level       = logging.INFO
        )

import torch
import torch.distributed as dist
from tqdm import tqdm

## own
from utils import (
    get_yaml_file,
    set_seed,
    get_linear_scheduler,
)
from model import DualEncoder, calculate_dpr_loss, LossCalculator
from data import QADataset, BEIRDataset

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/train_dpr_nq.yaml')
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

def calculate_hit_cnt(matching_score,labels):
    _, max_ids = torch.max(matching_score,1)
    return (max_ids == labels).sum()

def calculate_average_rank(matching_score,labels):
    _,indices = torch.sort(matching_score,dim=1,descending=True)
    ranks = []
    for idx,label in enumerate(labels):
        rank = ((indices[idx] == label).nonzero()).item() + 1  ##  rank starts from 1
        ranks.append(rank)
    return ranks

def validate(model,dataloader,accelerator):
    model.eval()
    query_embeddings = []
    positive_doc_embeddings = []
    negative_doc_embeddings = []
    for batch in dataloader:
        with torch.no_grad():
            query_embedding,doc_embedding  = model(**batch)
        query_num,_ = query_embedding.shape
        query_embeddings.append(query_embedding.cpu())
        positive_doc_embeddings.append(doc_embedding[:query_num,:].cpu())
        negative_doc_embeddings.append(doc_embedding[query_num:,:].cpu())
    
    query_embeddings = torch.cat(query_embeddings,dim=0)
    doc_embeddings = torch.cat(positive_doc_embeddings+negative_doc_embeddings,dim=0)
    matching_score = torch.matmul(query_embeddings,doc_embeddings.permute(1,0)) # bs, num_pos+num_neg
    labels = torch.arange(query_embeddings.shape[0],dtype=torch.int64).to(matching_score.device)
    loss = calculate_dpr_loss(matching_score,labels=labels).item()
    ranks = calculate_average_rank(matching_score,labels=labels)
    if accelerator.use_distributed and accelerator.num_processes>1:
        ranks_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(ranks_from_all_gpus,ranks)
        ranks = [x for y in ranks_from_all_gpus for x in y]

        loss_from_all_gpus = [None for _ in range(accelerator.num_processes)] 
        dist.all_gather_object(loss_from_all_gpus,loss)
        loss = sum(loss_from_all_gpus)/len(loss_from_all_gpus)
    
    return sum(ranks)/len(ranks),loss

def main():
    args = parse_args()
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    if args.vram_fraction != 1:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        print(f">>> set vram fraction to {args.vram_fraction*torch.cuda.get_device_properties(0).total_memory/1024**3}GB")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )
    if accelerator.is_local_main_process:
        LOG_DIR = os.path.join(args.log_dir, args.run_name)

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    query_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False)
    doc_encoder = BertModel.from_pretrained(args.base_model,add_pooling_layer=False)
    dual_encoder = DualEncoder(query_encoder,doc_encoder)
    dual_encoder.train()

    collate_fn = QADataset.collate_fn if "msmarco" not in args.run_name else BEIRDataset.collate_fn

    train_dataset = QADataset(args.train_file) if "msmarco" not in args.run_name else BEIRDataset(args.train_file)
    train_collate_fn = functools.partial(collate_fn,tokenizer=tokenizer,stage='train',args=args,hard_neg=args.use_hard_neg)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=train_collate_fn,num_workers=4,pin_memory=True)
    
    dev_dataset = QADataset(args.dev_file) if "msmarco" not in args.run_name else BEIRDataset(args.dev_file)
    dev_collate_fn = functools.partial(collate_fn,tokenizer=tokenizer,stage='dev',args=args,hard_neg=args.use_hard_neg)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=dev_collate_fn,num_workers=4,pin_memory=True)

    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in dual_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in dual_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_eps)
        
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps*accelerator.num_processes))
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    SAVE_EPOCHS = 20 if MAX_TRAIN_EPOCHS > 40 else 40
    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)
    loss_calculator = LossCalculator(args,hard_neg=args.use_hard_neg)

    dual_encoder, optimizer, train_dataloader, dev_dataloader, loss_calculator = accelerator.prepare(
        dual_encoder, optimizer, train_dataloader, dev_dataloader, loss_calculator
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for step,batch in enumerate(train_dataloader):
            with accelerator.accumulate(dual_encoder):
                with accelerator.autocast():
                    query_embedding,doc_embedding  = dual_encoder(**batch)
                    single_device_query_num,_ = query_embedding.shape
                    single_device_doc_num,_ = doc_embedding.shape
                    if accelerator.use_distributed:
                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)

                    if args.cont_cache :
                        loss = loss_calculator(query_embedding,doc_embedding)
                    else : 
                        matching_score = torch.matmul(query_embedding,doc_embedding.permute(1,0))
                        labels = torch.cat([torch.arange(single_device_query_num) + gpu_index * single_device_doc_num for gpu_index in range(accelerator.num_processes)],dim=0).to(matching_score.device)
                        loss = calculate_dpr_loss(matching_score,labels=labels)

                accelerator.backward(loss)

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    completed_steps += 1
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    accelerator.log({"training_loss": loss}, step=completed_steps)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0], "epoch" : epoch}, step=completed_steps)
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_calculator.empty_cache() # empty cache after one optimization step, if prev_cache is False

        if (epoch+1) % 10 == 0 :
            print(f"evaluating on dev set...")            
            avg_rank,loss = validate(dual_encoder,dev_dataloader,accelerator)
            dual_encoder.train()
            accelerator.log({"val_avg_rank": avg_rank, "val_loss":loss}, step=completed_steps)
            accelerator.wait_for_everyone()
            
            
    if accelerator.is_local_main_process:
        unwrapped_model = accelerator.unwrap_model(dual_encoder)
        unwrapped_model.query_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}_epoch-{epoch}/query_encoder"))
        tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}_epoch-{epoch}/query_encoder"))
        
        unwrapped_model.doc_encoder.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}_epoch-{epoch}/doc_encoder"))
        tokenizer.save_pretrained(os.path.join(LOG_DIR,f"step-{completed_steps}_epoch-{epoch}/doc_encoder"))

    accelerator.wait_for_everyone()
                

    
    accelerator.end_training()

if __name__ == '__main__':
    main()