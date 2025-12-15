from typing import List, Union, Tuple, Dict, Any
from pydantic import BaseModel

class DataSegmentConfig(BaseModel):
    name: str = "base"
    num_examples: int = 1_000
    input_seq_len: int = 64
    
    def build(self, seed: int):
        raise NotImplementedError

class DataConfig(BaseModel):
    train_configs: List[DataSegmentConfig]
    test_configs: List[DataSegmentConfig]
    batch_size: Union[int, Tuple[int, int]] = 32
    seed: int = 123
    cache_dir: str = None
    force_cache: bool = False