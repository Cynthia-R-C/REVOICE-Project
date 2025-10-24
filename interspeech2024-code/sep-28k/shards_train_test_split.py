# Cynthia Chen
# 10/19/2025
# Purpose: create train-validation-test split for generated shards from SEP-28k dataset for StutterNet training
# train - 80%, dev - 10%, test - 10%
# This file must be run from inside the interspeech2024-code folder

from pathlib import Path
import random

all_shards = [line.strip() for line in open('data/train/data.list') if line.strip()]
random.seed(42)
random.shuffle(all_shards)

n = len(all_shards)
train_split = int(0.8 * n)
dev_split = int(0.9 * n)

train_shards = all_shards[:train_split]
dev_shards = all_shards[train_split:dev_split]
test_shards = all_shards[dev_split:]

Path("data/train/data.list").write_text("\n".join(train_shards) + "\n", encoding="utf8")

Path("data/dev").mkdir(parents=True, exist_ok=True)
Path("data/dev/data.list").write_text("\n".join(dev_shards) + "\n", encoding="utf8")

Path("data/test").mkdir(parents=True, exist_ok=True)
Path("data/test/data.list").write_text("\n".join(test_shards) + "\n", encoding="utf8")
