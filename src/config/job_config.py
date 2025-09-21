from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Job:
    """Top-level job options and metadata."""

    # Optional path to a TOML config file (also used by ConfigManager for preloading)
    config_file: str | None = None


@dataclass(frozen=True)
class Data:
    train_bin: str = ""
    val_bin: str | None = None


@dataclass(frozen=True)
class Model:
    # Match defaults from GPTConfig / train.py
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = True
    # vocab_size is derived from dataset meta by default
    vocab_size: int | None = None


@dataclass(frozen=True)
class Init:
    # 'scratch' | 'resume' | 'gpt2' | 'gpt2-medium' | 'gpt2-large' | 'gpt2-xl'
    init_from: str = "scratch"


@dataclass(frozen=True)
class Optimizer:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0


@dataclass(frozen=True)
class LRScheduler:
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5


@dataclass(frozen=True)
class Training:
    max_iters: int = 600000
    gradient_accumulation_steps: int = 5 * 8
    batch_size: int = 12
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    seed: int = 1024
    always_save_checkpoint: bool = True


@dataclass(frozen=True)
class Distributed:
    backend: str = "nccl"  # 'nccl', 'gloo', etc.


@dataclass(frozen=True)
class System:
    device: str = "cuda"  # 'cpu', 'cuda', 'cuda:0', ...
    # 'auto' picks bfloat16 if supported, else float16; can be 'float32'|'bfloat16'|'float16'
    dtype: str = "auto"
    compile: bool = True


@dataclass(frozen=True)
class Logging:
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"
    wandb_group: str | None = None
    wandb_notes: str | None = None
    # Folders; manager will ensure they exist
    log_folder: str = "out"
    checkpoint_folder: str = "out"


@dataclass(frozen=True)
class JobConfig:
    """Configuration container for training."""

    job: Job = field(default_factory=Job)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    init: Init = field(default_factory=Init)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    distributed: Distributed = field(default_factory=Distributed)
    system: System = field(default_factory=System)
    logging: Logging = field(default_factory=Logging)

    def to_dict(self) -> dict[str, any]:  # pyright: ignore
        return asdict(self)
