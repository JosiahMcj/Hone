"""Prompt suite: fixed, hashed, frozen. Editing the JSON changes the hash
and voids comparisons (PROTOCOL frozen-artifacts law)."""
import hashlib
import json
import os

SUITE_T1 = os.path.join(os.path.dirname(__file__), os.pardir,
                        "suites", "t1", "prompts.json")

REQUIRED_KEYS = {"id", "category", "prompt", "max_tokens"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_suite(path=SUITE_T1):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt ids in suite")
    for it in items:
        missing = REQUIRED_KEYS - set(it)
        if missing:
            raise ValueError(f"prompt {it.get('id')} missing {missing}")
    return items, sha256_file(path)
