import torch
import random, json

from utils import normalize_query, normalize_document
class QADataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        data = json.load(open(file_path))
        self.data = [r for r in data if len(r["positive_ctxs"]) > 0 and (len(r["hard_negative_ctxs"]) > 0 or len(r["negative_ctxs"]) > 0)]

        print(f"""
>>> DATASET INFO
    - file_path: {file_path}
    - number of samples (before cleaning): {len(data)}
    - number of samples (after cleaning): {len(self.data)}
            """)    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples,tokenizer,args,stage,hard_neg=True):
        
        # prepare query input
        queries = [normalize_query(x['question']) for x in samples]
        query_inputs = tokenizer(queries,max_length=256,padding=True,truncation=True,return_tensors='pt')
        
        # prepare document input
        ## select the first positive document
        positive_passages = [x['positive_ctxs'][0] for x in samples]
        positive_titles = [x['title'] for x in positive_passages]
        positive_docs = [x['text'] for x in positive_passages]

        if stage == 'train':
            ## random choose one negative document
            negative_passages = [random.choice(x['hard_negative_ctxs']) 
                                if len(x['hard_negative_ctxs']) != 0  
                                else random.choice(x['negative_ctxs']) 
                                for x in samples ]
        elif stage == 'dev':
            negative_passages = [x['hard_negative_ctxs'][:min(args.num_hard_negative_ctx,len(x['hard_negative_ctxs']))]
                                + x['negative_ctxs'][:min(args.num_other_negative_ctx,len(x['negative_ctxs']))] 
                                for x in samples]
            negative_passages = [x for y in negative_passages for x in y]

        negative_titles = [x["title"] for x in negative_passages]
        negative_docs = [x["text"] for x in negative_passages]
        if hard_neg : 
            titles = positive_titles + negative_titles
            docs = positive_docs + negative_docs
        else:
            titles = positive_titles
            docs = positive_docs
        doc_inputs = tokenizer(titles,docs,max_length=256,padding=True,truncation=True,return_tensors='pt')

        return {
            'query_input_ids':query_inputs.input_ids,
            'query_attention_mask':query_inputs.attention_mask,
            'query_token_type_ids':query_inputs.token_type_ids,

            "doc_input_ids":doc_inputs.input_ids,
            "doc_attention_mask":doc_inputs.attention_mask,
            "doc_token_type_ids":doc_inputs.token_type_ids,
        }
    

class BEIRDataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.data = json.load(open(file_path))
        # self.data = [r for r in data if len(r["positive_ctxs"]) > 0 and (len(r["hard_negative_ctxs"]) > 0 or len(r["negative_ctxs"]) > 0)]

        print(f"""
>>> DATASET INFO
    - file_path: {file_path}
    - number of samples : {len(self.data)}
            """)    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples,tokenizer,args,stage,hard_neg=True):
        
        # prepare query input
        queries = [normalize_query(x['question']['text']) for x in samples]
        query_inputs = tokenizer(queries,max_length=256,padding=True,truncation=True,return_tensors='pt')
        
        # prepare document input
        ## select the first positive document
        ## passage = title + document
        positive_passages = [x['positive_ctxs'][0] for x in samples]
        positive_titles = [x['title'] for x in positive_passages]
        positive_docs = [x['text'] for x in positive_passages]

        if stage == 'train':
            ## random choose one negative document
            negative_passages = [random.choice(x['negative_ctxs']) 
                                for x in samples ]
            negative_titles = [x["title"] for x in negative_passages]
            negative_docs = [x["text"] for x in negative_passages]

        if hard_neg:
            titles = positive_titles + negative_titles if stage == 'train' else positive_titles
            docs = positive_docs + negative_docs if stage == 'train' else positive_docs
        else:
            titles = positive_titles
            docs = positive_docs
            
        doc_inputs = tokenizer(titles,docs,max_length=256,padding=True,truncation=True,return_tensors='pt')

        return {
            'query_input_ids':query_inputs.input_ids,
            'query_attention_mask':query_inputs.attention_mask,
            'query_token_type_ids':query_inputs.token_type_ids,

            "doc_input_ids":doc_inputs.input_ids,
            "doc_attention_mask":doc_inputs.attention_mask,
            "doc_token_type_ids":doc_inputs.token_type_ids,
        }