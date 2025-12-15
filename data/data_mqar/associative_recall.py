import numpy as np
import torch
from .config import DataSegmentConfig 
from .utils import DataSegment      

class MQARConfig(DataSegmentConfig):
    name: str="multiquery_ar"
    power_a: float=0.01
    num_kv_pairs: int=8
    random_non_queries: bool=True
    include_slices: bool=True

    def build(self, seed: int) -> DataSegment:
        return multiquery_ar(**self.model_dump(), seed=seed)

def multiquery_ar(vocab_size, num_examples, input_seq_len, seed, power_a=0.01, num_kv_pairs=8, random_non_queries=True, **kwargs):
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    
    np.random.seed(seed)
    context_size = num_kv_pairs * 2

    # Keys and Values
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys = np.apply_along_axis(np.random.choice, 1, np.tile(key_choices, (num_examples, 1)), replace=False, size=num_kv_pairs)
    values = np.apply_along_axis(np.random.choice, 1, np.tile(value_choices, (num_examples, 1)), replace=False, size=num_kv_pairs)

    # Interleave Keys and Values
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Power Law Gaps
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    
    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # Construct Queries
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    
    examples = np.concatenate([kvs, queries], axis=1)
    
    # Construct Labels
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
        
    return DataSegment(inputs, labels, slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len})