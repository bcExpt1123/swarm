#!/usr/bin/env python3
"""Quick check that you are running train_pretrain, not the TensorBoard server."""
import sys
from pathlib import Path

print("verify_train_cli: OK")
print("executable:", sys.executable)
script = Path(__file__).resolve().parent / "train_pretrain.py"
print("train_pretrain exists:", script.is_file())
print("first lines of train_pretrain.py:")
for i, line in enumerate(script.read_text(encoding="utf-8").splitlines()[:8]):
    print(f"  {i+1}: {line}")
