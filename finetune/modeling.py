import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    def __init__(self,
                 model_name: str = None,
                 normlized: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 train_batch_size: int = None,
                 ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self. train_batch_size = train_batch_size

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        else:
            raise ValueError(f"no match pooling method for {self.pooling_method}, available method is [`cls`,`mean`,`last`]")


    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = self.sentence_embedding(psg_out.last_hidden_state, features['attention_mask'])
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self, 
        query_pos: Dict[str, Tensor] = None, 
        query_neg: Dict[str, Tensor] = None, 
        passage_pos: Dict[str, Tensor] = None,
        passage_neg: Dict[str, Tensor] = None
        ):

        q_pos_reps = self.encode(query_pos)
        q_neg_reps = self.encode(query_neg)
        p_pos_reps = self.encode(passage_pos)
        p_neg_reps = self.encode(passage_neg)

        # print(f"before gather input batch is {q_reps.size(0)}")

        if self.negatives_cross_device and self.use_inbatch_neg:
            q_pos_reps = self._dist_gather_tensor(q_pos_reps)
            q_neg_reps = self._dist_gather_tensor(q_neg_reps)
            p_pos_reps = self._dist_gather_tensor(p_pos_reps)
            p_neg_reps = self._dist_gather_tensor(p_neg_reps)

        # print(f"after gather input batch is {q_reps.size(0)}")

        q_reps = torch.cat([q_pos_reps,q_neg_reps], dim=0).contiguous()
        p_reps = torch.cat([p_pos_reps,p_neg_reps,q_neg_reps], dim=0).contiguous()

        # q_reps = q_pos_reps.contiguous()
        # p_reps = q_pos_reps.contiguous()
        

        if p_reps.size(0)%q_reps.size(0) or p_reps.size(0)==q_reps.size(0):
            k,m = q_reps.size(0),p_reps.size(0)-q_reps.size(0)
            scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
            scores = scores.view(q_reps.size(0), -1)

            target = list(range(k-m)) + list(range(k,k+m))
            target = torch.tensor(target, device=scores.device, dtype=torch.long)

            # log_info = f"process_rank{self.process_rank}:"
            # for x,y in zip(p_reps[-1][-10:],q_reps[-1][-10:]):
            #     log_info+=f"\n{x.item()}\t{y.item()}"
            # log_info+=f"p_reps shape:{p_reps.size(0)},q_reps shape:{q_reps.size(0)}"

            # print(log_info)
            loss = self.compute_loss(scores, target)
        else:
            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t_size = t.size()
        t = torch.nn.functional.pad(t, pad=(0,0,0,self.train_batch_size*2-t_size[0]), value=float("nan")).contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t

        gather_tensors = torch.cat([x[~torch.isnan(x).all(dim=-1)] for x in all_tensors], dim=0)

        return gather_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)



