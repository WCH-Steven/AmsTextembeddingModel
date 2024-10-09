import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_format = args.data_format

        self._load_by_format(args)
        self.instruct_lookup = {
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

        self.total_len = len(self.dataset)

    def _load_by_format(self, args):
        if not args.data_format or args.data_format in ("BAAI", "knlin"):
            if os.path.isdir(args.train_data):
                train_datasets = []
                for file in os.listdir(args.train_data):
                    temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                        split='train')
                    if len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(train_datasets)
            else:
                train_datasets = []
                for file in args.train_data.split("??"):
                    temp_dataset = datasets.load_dataset('json', data_files=file,
                                                        split='train')
                    if len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_datasets.append(temp_dataset)
                self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            raise ValueError(
                f"Data format ({args.data_format}) not support for now, reformat data or check you format name is `knlin` `BAAI` or None"
            )

    def _truncate(self, text):
        seq_max_len = self.args.seq_max_len
        if len(text)<=seq_max_len:
            _text = text
        else:
            _text = text[:(seq_max_len-3) // 2] + '...' + text[-(seq_max_len-3) // 2:]
        return _text

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        if not self.data_format or self.data_format in ("BAAI"):
            query = self.dataset[item]['query']
            if self.args.query_instruction_for_retrieval is not None:
                query = self.args.query_instruction_for_retrieval + query

            passages = []

            assert isinstance(self.dataset[item]['pos'], list)
            pos = random.choice(self.dataset[item]['pos'])
            passages.append(pos)

            if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
                negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
            else:
                negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
            passages.extend(negs)

            if self.args.passage_instruction_for_retrieval is not None:
                passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
            return query, passages
        elif self.data_format in ("knlin"):
            src_text,dst_text = self.dataset[item]["pair"][0],self.dataset[item]["pair"][1]
            src_type,dst_type = self.dataset[item]["source"][0],self.dataset[item]["source"][1]
            src_instruct,dst_instruct = self.instruct_lookup.get(src_type, ""),self.instruct_lookup.get(dst_type, "")

            label = self.dataset[item]["label"]
            return {
                "src_text": self._truncate(src_instruct+src_text),
                "dst_text": self._truncate(dst_instruct+dst_text),
                "src_instruct": self._truncate(src_instruct),
                "dst_instruct": self._truncate(dst_instruct),
                "label": label
            }



@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    # def padding_score(self, teacher_score):
    #     group_size = None
    #     for scores in teacher_score:
    #         if scores is not None:
    #             group_size = len(scores)
    #             break
    #     if group_size is None:
    #         return None

    #     padding_scores = [100.0] + [0.0] * (group_size - 1)
    #     new_teacher_score = []
    #     for scores in teacher_score:
    #         if scores is None:
    #             new_teacher_score.append(padding_scores)
    #         else:
    #             new_teacher_score.append(scores)
    #     return new_teacher_score

    def __init__(self, tokenizer, data_args, training_args):
        self.data_args = data_args
        self.training_args = training_args
        self.data_format = data_args.data_format
        self.sentence_pooling_method = training_args.sentence_pooling_method
        self.tokenizer = tokenizer

    def _renew_mask(self, texts, instructs, use_instruct=None):
        texts = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.data_args.seq_max_len,
                return_tensors="pt",
            )
        if not use_instruct:
            return texts
        instructs = self.tokenizer(
                instructs,
                padding=True,
                truncation=True,
                max_length=self.data_args.seq_max_len,
                return_tensors="pt",
            )

        texts_attention_mask = texts["attention_mask"].clone()
        instructs_attention_mask = instructs["attention_mask"][:,1:].clone()

        texts_attention_mask[:,:instructs_attention_mask.size(1)] = texts_attention_mask[:,:instructs_attention_mask.size(1)] - instructs_attention_mask
        texts["attention_mask"] = texts_attention_mask.clone()

        return texts

    def __call__(self, features):
        if self.data_format in ("knlin"):
            src_texts,dst_texts = [],[]
            src_instructs,dst_instructs = [],[]
            m = 0
            for feature in features:
                src_text,dst_text = feature["src_instruct"]+feature["src_text"],feature["dst_instruct"]+feature["dst_text"]
                src_instruct,dst_instruct = feature["src_instruct"],feature["dst_instruct"]

                if feature["label"]:
                    m+=1
                    src_texts = [src_text] + src_texts
                    dst_texts = [dst_text] + dst_texts
                    src_instructs = [src_instruct] + src_instructs
                    dst_instructs = [dst_instruct] + dst_instructs
                else:
                    src_texts.append(src_text)
                    dst_texts.append(dst_text)
                    src_instructs.append(src_instruct)
                    dst_instructs.append(dst_instruct)


            q_pos_collated = self._renew_mask(src_texts[:m],src_instructs[:m],use_instruct=self.data_args.use_instruct)
            q_neg_collated = self._renew_mask(src_texts[m:],src_instructs[m:],use_instruct=self.data_args.use_instruct)
            d_pos_collated = self._renew_mask(dst_texts[:m],dst_instructs[:m],use_instruct=self.data_args.use_instruct)
            d_neg_collated = self._renew_mask(dst_texts[m:],dst_instructs[m:],use_instruct=self.data_args.use_instruct)
            return {
                "query_pos": q_pos_collated,
                "query_neg": q_neg_collated, 
                "passage_pos": d_pos_collated,
                "passage_neg": d_neg_collated
                }
        else:
            query = [f[0] for f in features]
            passage = [f[1] for f in features]

            if isinstance(query[0], list):
                query = sum(query, [])
            if isinstance(passage[0], list):
                passage = sum(passage, [])

            q_collated = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            d_collated = self.tokenizer(
                passage,
                padding=True,
                truncation=True,
                max_length=self.passage_max_len,
                return_tensors="pt",
            )
            return {"query": q_collated, "passage": d_collated}
