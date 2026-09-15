"""Multi-turn chat suite for T3-A (prompt-cache reuse). Separate from the
single-turn T1 suite loader on purpose (different shape, different frozen
hash). Each item is a fixed conversation prefix plus one final user turn;
the harness generates only the final assistant turn. Frozen + hashed like
the T1 suite (PROTOCOL frozen-artifacts law)."""
import os

from hone.suite import sha256_file  # reuse the identical hashing

SUITE_T3A = os.path.join(os.path.dirname(__file__), os.pardir,
                         "suites", "t3a", "chat.json")

REQUIRED_KEYS = {"id", "category", "prefix_messages", "final_user", "max_tokens"}


def load_chat_suite(path=SUITE_T3A):
    import json
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt ids in chat suite")
    for it in items:
        missing = REQUIRED_KEYS - set(it)
        if missing:
            raise ValueError(f"chat item {it.get('id')} missing {missing}")
        if not isinstance(it["prefix_messages"], list) or not it["prefix_messages"]:
            raise ValueError(f"chat item {it['id']}: prefix_messages must be a non-empty list")
        for m in it["prefix_messages"]:
            if set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"chat item {it['id']}: bad message {m!r}")
        if not isinstance(it["final_user"], str):
            raise ValueError(f"chat item {it['id']}: final_user must be a string")
    return items, sha256_file(path)
