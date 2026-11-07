"""Kaggle entrypoint: clone the repo at BRANCH and run CMD inside it. Outputs go to /kaggle/working."""
import os
import subprocess
import sys

REPO = "https://github.com/omnaiduu/RAG.git"
BRANCH = "claude/gallant-hamilton-rgm4j1"
PIP = []  # filled per job
CMD = []  # filled per job

subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "/kaggle/working/repo"], check=True)
subprocess.run(["git", "-C", "/kaggle/working/repo", "log", "-1", "--oneline"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
if PIP:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP], check=True)
env = {**os.environ, "PYTHONPATH": "/kaggle/working/repo", "HF_HOME": "/kaggle/working/hf-home",
       "TOKENIZERS_PARALLELISM": "false"}
rc = subprocess.run(CMD, cwd="/kaggle/working/repo", env=env).returncode
subprocess.run(["rm", "-rf", "/kaggle/working/repo", "/kaggle/working/hf-home"])
sys.exit(rc)
