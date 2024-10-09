import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
# import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm


class AmsTextModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name_or_path,
                 max_seq_length: int = 256,
                 sentence_pooling_method: str = "mean",
                 normlized: bool = True,
                 device: str = "cpu"
                 ):
        super().__init__()

        self.device = device
        self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_seq_length = max_seq_length
        self.sentence_pooling_method = sentence_pooling_method
        self.normlized = normlized
        self.name2instruct = {
            "公众号文章标题": "文章标题：",
            "兴趣类目": "用户兴趣：",
            "用户兴趣": "用户兴趣：",
            "商业类目": "商业类目：",
            "总结短语": "总结短语：",
            "搜索文本": "搜索文本：",
            "相关商品": "商品：",
            "商品":"商品：",
            "视频号summary": "视频简介：",
            "视频号标题": "视频标题："
        }
        

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'last':
            return hidden_state[:, -1]
        else:
            raise ValueError(f"no match pooling method for {self.pooling_method}, available method is [`cls`,`mean`,`last`]")

    def tokenize(self, text):
        tokenized =  self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        tokenized_ = {}
        for k in tokenized:
            tokenized_[k] = tokenized[k].to(self.model.device)
        return tokenized_


    def _encode(self, texts, instructs):
        texts = self.tokenize(texts)
        instructs = self.tokenize(instructs)
        model_out = self.model(**texts, return_dict=True)

        texts_attention_mask = texts["attention_mask"]
        instructs_attention_mask = instructs["attention_mask"]
        instructs_attention_mask = instructs_attention_mask[:,1:]

        texts_attention_mask[:,:instructs_attention_mask.size(1)] = texts_attention_mask[:,:instructs_attention_mask.size(1)] - instructs_attention_mask
        mask = texts_attention_mask
        # print(texts["attention_mask"],instructs["attention_mask"],mask)
        text_embs = self.sentence_embedding(model_out.last_hidden_state, mask)
        if self.normlized:
            text_embs = torch.nn.functional.normalize(text_embs, dim=-1)
        return text_embs.contiguous()


    # @torch.inference_mode
    def encode(self, texts, instructs, batch_size=1000):
        with torch.no_grad():
            instructs = [self.name2instruct.get(instructs[i], "") for i in range(len(instructs))]
            texts = [y+x for x,y in zip(texts,instructs)]
            text_embs = []
            for batch_id in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[batch_id:batch_id+batch_size]
                batch_instructs = instructs[batch_id:batch_id+batch_size]
                batch_embs = self._encode(batch_texts,batch_instructs)
                text_embs.append(batch_embs.cpu().detach())

            text_embs = torch.cat(text_embs, dim=0)

            return text_embs

def get_faiss_embedding(texts, instructs, model, batch_size=1000, return_type="np"):
    text_embs = []
    for batch_id in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[batch_id:batch_id+batch_size]
        batch_instructs = instructs[batch_id:batch_id+batch_size]
        batch_embs = model(batch_texts, batch_instructs)
        text_embs.append(batch_embs.cpu().detach())

    
    text_embs = torch.cat(text_embs, dim=0)
    if return_type=="np":
        text_embs = text_embs.numpy()

    return text_embs

if __name__ == "__main__":
    texts = ["你啊好"]*10000
    instructs = ["兴趣类目"]*10000

    model = AmsTextModel(
        model_name_or_path = "/cephfs/group/teg-openrecom-openrc/hainnwang/AmsNlp/source/save_models/sph_325_98_mean_sl96_instruct0_t005_epoch2_bs512",
        sentence_pooling_method = "mean",
        normlized = True,
        device = "cuda:0"
    )

    embeddings = model.encode(texts=texts, instructs=instructs, batch_size=2500)
    print(embeddings.shape)
    print(embeddings[0])

