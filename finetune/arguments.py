import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/apdcephfs_qy3/share_1603729/hainnwang/search/source/bge-large-zh-v1.5",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )



@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )

    train_group_size: int = field(default=8)

    data_format: str = field(
        default="knlin",
        metadata={
            "help": "load data by data format, now supports bot `knlin` and `BAAI`"
        }
    )

    use_instruct: bool = field(
        default=True,
        metadata={
            "help": "whether to use instruct for text "
        }
    )

    seq_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for texts. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            for file in self.train_data.split("??"):
                if not os.path.exists(file):
                    raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.06)
    fix_position_embedding: bool = field(default=True, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean or mean_including_cls"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
