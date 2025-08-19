from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class TrainConfig:
    seed: int = 1337
    threads: int = 4
    batch_size: int = 32
    microbatch: int = 1
    grad_accum: int = 64
    grad_clip: float = 1.0
    dtype: str = "float32"
    eval_every: int = 100
    save_every: int = 200


@dataclass
class OptimConfig:
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)


@dataclass
class ScheduleConfig:
    type: str = "cosine"
    warmup_steps: int = 50


@dataclass
class ModelConfig:
    model: str = "gpt_mini"
    vocab_size: int = 128
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    mlp_ratio: int = 4
    rope: bool = True
    norm: str = "rms"


@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    train: TrainConfig = TrainConfig()


def load_yaml_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = Config()

    if "model" in data:
        for k, v in data["model"].items():
            setattr(cfg.model, k, v)
    if "optim" in data:
        for k, v in data["optim"].items():
            setattr(cfg.optim, k, v)
    if "schedule" in data:
        for k, v in data["schedule"].items():
            setattr(cfg.schedule, k, v)
    if "train" in data:
        for k, v in data["train"].items():
            setattr(cfg.train, k, v)
    return cfg


