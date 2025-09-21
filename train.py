"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Configuration is read from `config/job_config.py` dataclasses via `ConfigManager`.

To run on a single GPU, example:
$ python train.py --training.batch_size=32 --system.compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import json
import math
import os
import shutil
import time
import uuid
import glob
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Optional, cast

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.job_config import JobConfig, Model
from src.config.manager import ConfigManager
from src.experiment import cfg_hash, run_dir, write_meta
from src.model import GPT


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)  # pyright: ignore[reportOptionalOperand]
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


def main(config: JobConfig) -> None:
    # -----------------------------------------------------------------------------
    # Unpack config into local variables (matches original script expectations)
    # wandb logging
    wandb_log = config.logging.wandb_log
    wandb_project = config.logging.wandb_project
    wandb_run_name = config.logging.wandb_run_name
    wandb_group = config.logging.wandb_group
    wandb_notes = config.logging.wandb_notes
    # I/O: structured run directory
    run_id = os.environ.get("RUN_ID", uuid.uuid4().hex[:8])
    out_dir = run_dir(
        config.logging.log_folder,
        wandb_project,
        wandb_group,
        wandb_run_name,
        config,
        run_id,
    )
    eval_interval = config.training.eval_interval
    log_interval = config.training.log_interval
    eval_iters = config.training.eval_iters
    eval_only = config.training.eval_only
    always_save_checkpoint = config.training.always_save_checkpoint
    init_from = config.init.init_from
    # data
    train_bin = config.data.train_bin
    val_bin = config.data.val_bin
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    batch_size = config.training.batch_size
    block_size = config.model.block_size
    # model
    dropout = config.model.dropout
    # adamw optimizer
    learning_rate = config.optimizer.learning_rate
    max_iters = config.training.max_iters
    weight_decay = config.optimizer.weight_decay
    beta1 = config.optimizer.beta1
    beta2 = config.optimizer.beta2
    grad_clip = config.optimizer.grad_clip
    # learning rate decay settings
    decay_lr = config.lr.decay_lr
    warmup_iters = config.lr.warmup_iters
    lr_decay_iters = config.lr.lr_decay_iters
    min_lr = config.lr.min_lr
    # DDP settings
    backend = config.distributed.backend
    # system
    device = config.system.device
    if config.system.dtype == "auto":
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    else:
        dtype = config.system.dtype
    compile = config.system.compile
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_rank = None
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        write_meta(out_dir, config)
    torch.manual_seed(config.training.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model: torch.nn.Module
    checkpoint: Optional[dict[str, Any]] = None
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        vocab = config.model.vocab_size
        if vocab is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )
        gptconf = Model(
            **{
                **asdict(config.model),
                "vocab_size": vocab,
            }
        )
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = cast(dict[str, Any], torch.load(ckpt_path, map_location=device))
        checkpoint_model_args: dict[str, Any] = checkpoint["model_args"]
        # create the model from checkpoint args
        gptconf = Model(**checkpoint_model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # convert torch.compile state dict back to regular state dict
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
    else:
        raise ValueError(f"Unknown init_from setting: {init_from}")
    # crop down the model block size if desired, using model surgery
    if isinstance(model, GPT) and block_size < model.config.block_size:
        model.crop_block_size(block_size)
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])  # pyright: ignore[reportArgumentType]
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        model = cast(torch.nn.Module, torch.compile(model))  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # pyright: ignore[reportPossiblyUnboundVariable]

    train_loader = DistributedDataLoader(
        train_bin,
        batch_size,
        block_size,
        ddp_rank or 0,
        ddp_world_size,
    )
    val_loader = None
    if val_bin:
        val_loader = DistributedDataLoader(
            val_bin,
            batch_size,
            block_size,
            ddp_rank or 0,
            ddp_world_size,
        )

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        if val_loader is None:
            return None
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = val_loader.next_batch()
                X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            notes=wandb_notes,
            config={
                "cfg_hash": cfg_hash(config),
                "run_id": run_id,
                "out_dir": out_dir,
                **asdict(config),
            },
            dir=out_dir,
        )

    # training loop
    X, Y = train_loader.next_batch()  # fetch the very first batch
    X, Y = X.to(device), Y.to(device)
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model: GPT = (
        cast(GPT, model.module) if ddp else cast(GPT, model)
    )  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = None
            if eval_iters > 0 and val_loader is not None:
                losses = cast(dict[str, torch.Tensor], estimate_loss())
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if wandb_log and master_process:
                    import wandb

                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )
            if (
                losses is not None and losses["val"] < best_val_loss
            ) or always_save_checkpoint:
                if losses is not None:
                    best_val_loss = losses["val"]
                if iter_num > 0:
                    ck_root = os.path.join(out_dir, "checkpoints")
                    step_dir = os.path.join(ck_root, f"step-{iter_num:06d}")
                    os.makedirs(step_dir, exist_ok=True)
                    torch.save(
                        raw_model.state_dict(), os.path.join(step_dir, "model.pt")
                    )
                    torch.save(
                        optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt")
                    )
                    with open(os.path.join(step_dir, "trainer_state.json"), "w") as f:
                        json.dump(
                            {
                                "iter_num": iter_num,
                                "best_val_loss": float(best_val_loss),
                            },
                            f,
                        )
                    latest = os.path.join(ck_root, "latest")
                    try:
                        if os.path.islink(latest) or os.path.exists(latest):
                            if os.path.islink(latest):
                                os.unlink(latest)
                            else:
                                shutil.rmtree(latest)
                        os.symlink(os.path.basename(step_dir), latest)
                    except Exception:
                        pass
                    print(f"saving checkpoint to {step_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        last_loss: Optional[torch.Tensor] = None
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                cast(DDP, model).require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            last_loss = loss
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = train_loader.next_batch()
            X, Y = X.to(device), Y.to(device)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = cast(torch.Tensor, last_loss).item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
            if wandb_log and master_process:
                import wandb

                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "mfu": running_mfu * 100,
                        "time_ms": dt * 1000,
                    }
                )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    main(config)
