from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualEncoder(nn.Module):
    def __init__(self,query_encoder,doc_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder

    def forward(
        self,
        query_input_ids, # [bs,seq_len]
        query_attention_mask, # [bs,seq_len]
        query_token_type_ids, # [bs,seq_len],
        doc_input_ids, # [bs*n_doc,seq_len]
        doc_attention_mask, # [bs*n_doc,seq_len]
        doc_token_type_ids, # [bs*n_doc,seq_len]
    ):  
        CLS_POS = 0
        ## [bs,n_dim]
        query_embedding = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask = query_attention_mask,
            token_type_ids = query_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        ## [bs * n_doc,n_dim]
        doc_embedding = self.doc_encoder(
            input_ids = doc_input_ids,
            attention_mask = doc_attention_mask,
            token_type_ids = doc_token_type_ids,
            ).last_hidden_state[:,CLS_POS,:]
        
        return query_embedding,doc_embedding 

def calculate_dpr_loss(matching_score,labels):
    return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)

class LossCalculator(nn.Module):
    def __init__(self, args, hard_neg=True):
        super().__init__()
        self.prev_cache = args.prev_cache
        self.cache_hard_neg = args.cache_hard_neg
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.cache_query = args.cache_query
        self.hard_neg = hard_neg

        if self.cache_query:
            self.query_cache = deque(maxlen=args.cache_size*self.gradient_accumulation_steps)
        else :
            self.query_cache = None
        self.neg_doc_cache = deque(maxlen=args.cache_size*self.gradient_accumulation_steps)
        if self.cache_hard_neg:
            self.hard_neg_doc_cache = deque(maxlen=args.cache_size*self.gradient_accumulation_steps)
        else :
            self.hard_neg_doc_cache = None

    def forward(self,query_embedding, doc_embedding):
        """
        Args:
            query_embedding : [bs,n_dim]
            doc_embedding : [bs*2,n_dim] (positive + negative)
        """
        # concat with cache
        query_embedding = self.concat_with_cache(query_embedding,self.query_cache)
        if self.hard_neg:
            neg_doc_embedding, hard_neg_doc_embedding = self.split_hard_neg_doc(doc_embedding)
            neg_doc_embedding = self.concat_with_cache(neg_doc_embedding,self.neg_doc_cache)
            hard_neg_doc_embedding = self.concat_with_cache(hard_neg_doc_embedding,self.hard_neg_doc_cache)
            doc_embedding = torch.cat([neg_doc_embedding,hard_neg_doc_embedding],dim=0)
        else:
            neg_doc_embedding = self.concat_with_cache(doc_embedding,self.neg_doc_cache)
            doc_embedding = neg_doc_embedding
             
        len_query = query_embedding.shape[0]
        labels = torch.arange(len_query).to(query_embedding.device) 
        matching_score = torch.matmul(query_embedding,doc_embedding.permute(1,0))
        loss = calculate_dpr_loss(matching_score,labels=labels)
        return loss
    
    def concat_with_cache(self,embedding,cache):        
        if type(cache) != deque :
            return embedding
        if len(cache) == 0:
            self.enque(embedding,cache)
            return embedding
        else:
            embedding_with_cache = torch.cat([embedding,torch.cat(list(cache)).to(embedding.device)],dim=0)
            self.enque(embedding,cache)
            return embedding_with_cache
        
    def enque(self, embedding, cache):
        if cache is None:
            return embedding
        else:
            cache.append(embedding.detach().clone())
    
    def split_hard_neg_doc(self,doc_embedding):
        neg_doc_embedding = doc_embedding[:len(doc_embedding)//2]
        hard_neg_doc_embedding = doc_embedding[len(doc_embedding)//2:]
        return neg_doc_embedding,hard_neg_doc_embedding
        
    def empty_cache(self):
        if not self.prev_cache and self.cont_cache:
            if self.query_cache :
                self.query_cache.clear()
            self.neg_doc_cache.clear()
            if self.cache_hard_neg:
                self.hard_neg_doc_cache.clear()
