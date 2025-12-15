import os 
import hashlib
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import numpy as np
import torch 
from torch.utils.data import DataLoader, Dataset
from .config import DataConfig, DataSegmentConfig 

@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: Dict[str, any] = None

    def __len__(self):
        return len(self.inputs)

    @classmethod
    def from_config(cls, config: DataSegmentConfig, cache_dir: str = None, force_cache: bool = False, seed: int = 123):
        def _get_cache_path(config: DataSegmentConfig):
            if cache_dir is None: return None
            # create hash based on config and seed
            config_hash = hashlib.md5(
                json.dumps({**config.model_dump(), "_seed": seed}, sort_keys=True).encode()
            ).hexdigest()
            return os.path.join(cache_dir, f"data_{config.name}_{config_hash}.pt")
        
        if cache_dir is not None:
            Path(cache_dir).mkdir(exist_ok=True, parents=True)
            
        cache_path = _get_cache_path(config)

        if cache_dir is not None and os.path.exists(cache_path) and not force_cache:
            print(f"Loading data from cache: {cache_path}") 
            try:
                return cls(**torch.load(cache_path))
            except RuntimeError:
                pass # Fallback to generation on error

        print(f"Generating dataset for {config.name}...") 
        data: DataSegment = config.build(seed=seed)

        if cache_dir is not None:
            print(f"Caching dataset to {cache_path}...") 
            torch.save(asdict(data), cache_path)
        return data

class _SyntheticDataset(Dataset):
    def __init__(self, segments: List[DataSegment], batch_size: int):
        self.segments = segments
        self.batch_size = batch_size        
        self.batches = [
            (segment_idx, batch_start)
            for segment_idx, segment in enumerate(self.segments)
            for batch_start in range(0, len(segment), self.batch_size)
        ]

    def __getitem__(self, batch_idx: int):
        segment_idx, batch_start = self.batches[batch_idx]
        segment = self.segments[segment_idx]
        slc = slice(batch_start, batch_start + self.batch_size)
        slices = [segment.slices if segment.slices is not None else {}] * self.batch_size
        return segment.inputs[slc], segment.labels[slc], slices      

    def __len__(self):
        return len(self.batches)

def prepare_data(config: DataConfig) -> Tuple[DataLoader, DataLoader]:  
    if isinstance(config.batch_size, int):
        train_bs, test_bs = (config.batch_size, config.batch_size)
    else:
        train_bs, test_bs = config.batch_size
    
    MAX_SEED = 2 ** 32
    np.random.seed(config.seed)
    # Generate distinct seeds for train vs test chunks
    train_seeds = np.random.randint(0, MAX_SEED // 2, size=len(config.train_configs))
    test_seeds = np.random.randint(MAX_SEED // 2, MAX_SEED, size=len(config.test_configs))
    
    kwargs = {"cache_dir": config.cache_dir, "force_cache": config.force_cache}
    
    train_ds = _SyntheticDataset([
        DataSegment.from_config(c, seed=int(s), **kwargs) for c, s in zip(config.train_configs, train_seeds)
    ], batch_size=train_bs)
    
    test_ds = _SyntheticDataset([
        DataSegment.from_config(c, seed=int(s), **kwargs) for c, s in zip(config.test_configs, test_seeds)
    ], batch_size=test_bs)

    # num_workers=0 is safer for synthetic data to avoid fork overhead on small batches
    return (
        DataLoader(train_ds, batch_size=None, num_workers=0, shuffle=False),
        DataLoader(test_ds, batch_size=None, num_workers=0, shuffle=False)
    )