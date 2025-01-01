from transformers.models.gpt2 import GPT2Tokenizer
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


from .compiled_args import DataTrainingArguments
from .data_utils.data_collator import DataCollatorForData2TextLanguageModeling
from .data_utils.language_modeling import LineByLineE2ETextDataset, LineByLineTriplesTextDataset



InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

def _tensorize_batch(
        examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", cache_dir="cache_distilgpt2")
tokenizer.padding_side = "left"
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"

data_collator = DataCollatorForData2TextLanguageModeling(
                tokenizer=tokenizer, mlm=False, format_mode="cat"
            )

_tensorize_batch(examples)