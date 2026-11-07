"""Kaggle entrypoint: clone the repo at BRANCH and run CMD inside it. Outputs go to /kaggle/working."""
import os
import subprocess
import sys

REPO = "https://github.com/omnaiduu/RAG.git"
BRANCH = "claude/gallant-hamilton-rgm4j1"
PIP = ['-U', 'transformers>=4.57', 'accelerate', 'bitsandbytes', 'bm25s', 'sentence-transformers', 'lancedb==0.41.0']
CMD = ['env', 'MODEL=google/gemma-4-E4B-it', 'GEMMA_QUANT=8bit', 'RACERS=C1v3,C2v2', 'SEARCH=lance', 'TAG=e4b-lance', 'bash', 'scripts/kaggle_small_racers.sh']

subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "/kaggle/working/repo"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/repo", "log", "-1", "--oneline"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
if PIP:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP], check=True)
env = {**os.environ, "PYTHONPATH": "/kaggle/working/repo", "HF_HOME": "/tmp/hf-home",
       "TOKENIZERS_PARALLELISM": "false"}
rc = subprocess.run(CMD, cwd="/kaggle/working/repo", env=env).returncode
subprocess.run(["rm", "-rf", "/kaggle/working/repo",])
sys.exit(rc)
