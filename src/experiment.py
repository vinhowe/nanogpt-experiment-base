import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict

from .config.job_config import JobConfig


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", s.strip())[:80].strip("-_.")


def _canonical_cfg_dict(cfg: JobConfig) -> dict:
    d = asdict(cfg)
    return {
        "model": d["model"],
        "data": d["data"],
        "optimizer": d["optimizer"],
        "lr": d["lr"],
        "seed": d["training"]["seed"],
    }


def cfg_hash(cfg: JobConfig) -> str:
    s = json.dumps(_canonical_cfg_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2s(s.encode(), digest_size=4).hexdigest()


def _shortsha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "nogit"


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def run_dir(
    root: str, project: str, group: str | None, name: str, cfg: JobConfig, runid: str
) -> str:
    return os.path.join(
        root,
        _slug(project),
        _slug(group or "default"),
        f"{_timestamp()}__{_slug(name)}__{cfg_hash(cfg)}__s{cfg.training.seed}__{_shortsha()}__{runid}",
    )


def write_meta(out_dir: str, cfg: JobConfig) -> None:
    import tomli_w

    meta = os.path.join(out_dir, "meta")
    os.makedirs(meta, exist_ok=True)

    with open(os.path.join(meta, "config.toml"), "w") as f:
        f.write(tomli_w.dumps(_canonical_cfg_dict(cfg)))

    # environment freeze (prefer uv)
    try:
        pf = subprocess.check_output(["uv", "pip", "freeze"]).decode()
    except Exception:
        pf = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    with open(os.path.join(meta, "pip_freeze.txt"), "w") as f:
        f.write(pf)

    # git commit
    full = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    with open(os.path.join(meta, "git_commit.txt"), "w") as f:
        f.write(full + "\n")

    # git dirtiness
    diff = subprocess.check_output(["git", "diff", "HEAD"]).decode()
    with open(os.path.join(meta, "git_diff.patch"), "w") as f:
        f.write(diff)
