import json
import shutil

import pytest

from hone.suite import SUITE_T1, load_suite


def test_frozen_suite_loads_and_hash_is_stable():
    items, sha1 = load_suite(SUITE_T1)
    _, sha2 = load_suite(SUITE_T1)
    assert len(items) == 12
    assert sha1 == sha2
    cats = {it["category"] for it in items}
    assert cats == {"chat", "code", "long"}


def test_canary_editing_suite_changes_hash(tmp_path):
    copy = tmp_path / "prompts.json"
    shutil.copy(SUITE_T1, copy)
    _, sha_orig = load_suite(str(copy))
    items = json.loads(copy.read_text())
    items[0]["prompt"] += " subtly gamed"
    copy.write_text(json.dumps(items, indent=1))
    _, sha_edited = load_suite(str(copy))
    assert sha_orig != sha_edited


def test_duplicate_ids_rejected(tmp_path):
    copy = tmp_path / "prompts.json"
    items = json.loads(open(SUITE_T1).read())
    items[1]["id"] = items[0]["id"]
    copy.write_text(json.dumps(items))
    with pytest.raises(ValueError, match="duplicate"):
        load_suite(str(copy))


def test_missing_keys_rejected(tmp_path):
    copy = tmp_path / "prompts.json"
    items = json.loads(open(SUITE_T1).read())
    del items[0]["max_tokens"]
    copy.write_text(json.dumps(items))
    with pytest.raises(ValueError, match="missing"):
        load_suite(str(copy))
