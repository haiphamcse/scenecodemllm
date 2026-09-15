"""Static + dry checks of each fork's eval.py (CPU, no model load).

Per fork: ast.parse, `eval.py --help` in a subprocess (exercises every import), then
import the module, check load_model/evaluate exist, and feed the flags of the fork's
eval slurm (env vars replaced by their `:-default` or a dummy) to its parse_args().

  python scripts/audit/check_eval_py.py            # all forks below
  python scripts/audit/check_eval_py.py FORK ...   # subset
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORKS = {  # fork -> eval slurm (None: no eval slurm yet)
    "idea_3i_130k_joint_cached": "scripts/idea_3i_130k_joint_cached/eval_full_v100.slurm",
    "idea_3i_130k_joint_cached_masked": "scripts/idea_3i_130k_joint_cached/eval_full_v100.slurm",
    "idea_3i_130k_joint_cached_64f": "scripts/idea_3i_130k_joint_cached_64f/eval_full_v100_64f.slurm",
    "idea_3i_130k_cached_64f": "scripts/idea_3i_130k_cached_64f/eval_full_v100.slurm",
    "idea_3i_130k_joint_cached_64f_4b": "scripts/idea_3i_130k_joint_cached_64f_4b/eval_full_v100_64f_4b.slurm",
    "idea_3i_130k_joint_cached_64f_vlm3r": "scripts/idea_3i_130k_joint_cached_64f_vlm3r/eval_full_v100_64f_vlm3r.slurm",
    "idea_3i_180k_cached_64f_lat512": None,
    "idea_3i_590k_cached": "scripts/idea_3i_590k_joint/eval_full_v100_cached_from1175.slurm",
    "idea_3i_590k_joint": "scripts/idea_3i_590k_joint/eval_full_v100.slurm",
}
_ENV = re.compile(r'"?\$\{(\w+)(?::-([^}]*))?\}"?')


def slurm_flags(path):
    """Tokens of the `args=( ... )` block, plus the conditional --max_samples line.

    `${VAR}` resolves to the slurm's own `VAR="${VAR:-default}"` line when there is one
    (SPLIT, DTYPE, ...), else to "1" (valid for every int/float/path flag)."""
    txt = (ROOT / path).read_text()
    defaults = dict(re.findall(r'^(\w+)="\$\{\1:-([^$}]*)\}"', txt, re.M))
    body = re.search(r"args=\((.*?)\n\)", txt, re.S).group(1)
    toks = []
    for line in body.splitlines():
        line = line.split("#")[0].strip()
        if line:
            toks += line.split()
    if "--max_samples" in txt:
        toks += ["--max_samples", "5"]
    sub = lambda m: m.group(2) if m.group(2) is not None else defaults.get(m.group(1), "1")
    return [_ENV.sub(sub, t).strip('"') for t in toks]


_PROBE = r"""
import importlib.util, sys
fork, flags = sys.argv[1], sys.argv[2:]
spec = importlib.util.spec_from_file_location("fork_eval", f"{fork}/eval.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for fn in ("load_model", "evaluate", "parse_args"):
    assert hasattr(m, fn), f"missing {fn}"
print("live_model_module:", m.Qwen3VLWithVggtForConditionalGeneration.__module__)
if flags:
    sys.argv = ["eval.py"] + flags
    a = m.parse_args()
    print("parsed:", {k: v for k, v in vars(a).items()
                     if k in ("frame_num_latents", "camera_num_latents", "frame_widening_factor",
                              "vggt_image_resolution", "video_fps", "video_max_frames", "model_path",
                              "num_shards", "shard_index", "max_samples", "dtype", "vggt_checkpoint")})
"""


def check(fork, slurm):
    ev = ROOT / fork / "eval.py"
    ast.parse(ev.read_text())
    env = dict(os.environ, PYTHONPATH=str(ROOT / fork), HF_HUB_OFFLINE="1")
    h = subprocess.run([sys.executable, str(ev), "--help"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    if h.returncode != 0:
        return f"BROKEN (--help): {h.stderr.strip().splitlines()[-1]}"
    flags = slurm_flags(slurm) if slurm else []
    p = subprocess.run([sys.executable, "-c", _PROBE, fork] + flags, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return f"BROKEN (parse): {p.stderr.strip().splitlines()[-1]}"
    return "OK " + " | ".join(p.stdout.strip().splitlines()) + (" | no eval slurm" if not slurm else "")


if __name__ == "__main__":
    for fork in sys.argv[1:] or FORKS:
        print(f"{fork}: {check(fork, FORKS[fork])}", flush=True)
