import os
import re
import json
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset, DatasetDict, concatenate_datasets, Image, Sequence, load_from_disk
from translator import Translator

API_TIMEOUT = 120
MAX_WORKERS = int(os.getenv("TX_WORKERS", "16"))  # tune via env var

translator = Translator(
    source_language="en",
    target_language="bg",
    timeout=API_TIMEOUT,
)

# one global pool reused across the whole run
_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

def _tx_one(text):
    return {"translation_text": translator.translate(text)}

def translate_batch(texts):
    # run individual translate() calls concurrently
    if isinstance(texts, list):
        return list(_pool.map(_tx_one, texts))  # preserves order
    else:
        return [{"translation_text": translator.translate(texts)}]

def _sanitize(s, default="default"):
    val = str(s or default)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", val)

def _ckpt_root(ds_meta):
    # Use dataset id + CONFIG NAME to separate checkpoints
    dataset_id = _sanitize(ds_meta["url"].split("/")[-1])
    config_name = _sanitize(ds_meta.get("name") or ds_meta.get("config") or "default")
    return f"./ckpt_{dataset_id}__{config_name}"

def _shard_dir(root, split, start, end):
    return os.path.join(root, split, f"{start:06d}-{end:06d}")

def _done_marker(dirpath):
    return os.path.join(dirpath, "_SUCCESS")

def _save_checkpoint_atomically(dataset_obj, final_dir):
    parent = os.path.dirname(final_dir)
    os.makedirs(parent, exist_ok=True)
    tmp_dir = final_dir + ".tmp"

    # clean up any stale tmp or partial dirs
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.exists(final_dir) and not os.path.exists(_done_marker(final_dir)):
        shutil.rmtree(final_dir, ignore_errors=True)

    # save to tmp then atomically rename
    dataset_obj.save_to_disk(tmp_dir)
    os.replace(tmp_dir, final_dir)
    with open(_done_marker(final_dir), "w") as f:
        json.dump({"saved_at": time.time()}, f)

def translate_dataset(
    ds, output_dataset_url,
    batch_size=32, hf_token_env='HF_TOKEN',
    checkpoint_interval=10, exclude_columns=None, preprocess_dataset=None,
):
    exclude_columns = exclude_columns or []

    has_hf_token = hf_token_env in os.environ  # <- fixed
    if not has_hf_token:
        print("[WARNING] Environment variable 'HF_TOKEN' is not set! The dataset will be translated but not uploaded to HuggingFace.")

    print("Loading dataset...")
    dataset = load_dataset(
        ds["url"],
        name=ds.get("name"),
        streaming=ds.get("streaming", False),
        config=ds.get("config"),
    )

    ckpt_root = _ckpt_root(ds)

    def batch_translate(batch):
        out = {}
        for col, data in batch.items():
            if col in exclude_columns:
                out[col] = data
                continue

            # translate strings
            if isinstance(data[0], str) and not data[0].isdigit():
                res = translate_batch(data)
                out[col] = [r["translation_text"] for r in res]

            # translate list[str]
            elif isinstance(data[0], list) and data[0] and isinstance(data[0][0], str):
                new_col = []
                for sub in data:
                    res = translate_batch(sub)
                    new_col.append([r["translation_text"] for r in res])
                out[col] = new_col

            # translate list[dict] (e.g., conversation turns)
            elif isinstance(data[0], list) and data[0] and isinstance(data[0][0], dict):
                new_col = []
                for sublist in data:
                    nd_sub = []
                    keys_to_tx, vals_to_tx = [], []
                    for i, d in enumerate(sublist):
                        nd = dict(d)
                        for k, v in nd.items():
                            if isinstance(v, str) and not v.isdigit():
                                keys_to_tx.append((i, k))
                                vals_to_tx.append(v)
                        nd_sub.append(nd)
                    if vals_to_tx:
                        res = translate_batch(vals_to_tx)
                        for (i, k), r in zip(keys_to_tx, res):
                            nd_sub[i][k] = r["translation_text"]
                    new_col.append(nd_sub)
                out[col] = new_col

            else:
                out[col] = data
        return out

    translated_splits = {}

    try:
        for split in dataset.keys():
            if preprocess_dataset is not None:
                dataset = preprocess_dataset(dataset, split)

            print(f"Processing split: {split}")
            ds_split = dataset[split]
            n = len(ds_split)

            shards = []
            for start in range(0, n, checkpoint_interval):
                end = min(start + checkpoint_interval, n)
                shard_path = _shard_dir(ckpt_root, split, start, end)
                done = _done_marker(shard_path)

                # If a complete checkpoint exists, load and skip work
                if os.path.exists(done):
                    print(f"Found completed checkpoint {split} {start}:{end}; loading.")
                    shard = load_from_disk(shard_path)
                    shards.append(shard)
                    continue

                # If a partial/incomplete checkpoint exists, clean it and recompute
                if os.path.exists(shard_path) and not os.path.exists(done):
                    print(f"Removing incomplete checkpoint at {shard_path} and recomputing.")
                    shutil.rmtree(shard_path, ignore_errors=True)

                # Compute shard
                print(f"Translating shard {start}:{end}...")
                shard = ds_split.select(range(start, end)).map(
                    batch_translate, batched=True, batch_size=batch_size
                )

                # Save atomically with a success marker
                _save_checkpoint_atomically(shard, shard_path)
                print(f"Shard {start}:{end} processed and checkpointed.")

                shards.append(shard)

            # Glue shards back together
            split_ds = concatenate_datasets(shards)

            # Fix image feature type if present
            if "images" in split_ds.column_names:
                split_ds = split_ds.cast_column("images", Sequence(Image()))

            translated_splits[split] = split_ds

    finally:
        # Always close the pool, even on errors
        _pool.shutdown(wait=True)

    if translated_splits:
        out = DatasetDict(translated_splits)
        # make sure we use the same config name that keyed the checkpoints
        config_name = (_sanitize(ds.get("name")) or _sanitize(ds.get("config")) or "default")
        if has_hf_token:
            out.push_to_hub(output_dataset_url, config_name=config_name)
            print(f"Pushed to {output_dataset_url} (config: {config_name})")
        else:
            print("Token not set: created translated dataset locally but did not upload.")
    else:
        print("No data was translated or saved.")
