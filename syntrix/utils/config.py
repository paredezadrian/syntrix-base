from dataclasses import dataclass, field
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
    train_steps: int = 300
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
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_yaml_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = Config()

    # Model section can be a string (name) or a dict
    if "model" in data:
        if isinstance(data["model"], str):
            cfg.model.model = data["model"]
        elif isinstance(data["model"], dict):
            for k, v in data["model"].items():
                setattr(cfg.model, k, v)
    # Support flattened top-level fields for model too
    for key in (
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "d_model",
        "mlp_ratio",
        "rope",
        "norm",
    ):
        if key in data:
            setattr(cfg.model, key, data[key])
    if "optim" in data and isinstance(data["optim"], dict):
        for k, v in data["optim"].items():
            setattr(cfg.optim, k, v)
    if "schedule" in data and isinstance(data["schedule"], dict):
        for k, v in data["schedule"].items():
            setattr(cfg.schedule, k, v)
    if "train" in data and isinstance(data["train"], dict):
        for k, v in data["train"].items():
            setattr(cfg.train, k, v)
    return cfg
