#!/usr/bin/env python

# python tokenizers/expand.py --out_dir tokenizers/Qwen3-VL-2B-Instruct_expand

import os
import argparse, os, re, json, logging
from collections import Counter
from typing import Iterable, List
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "1"

CYR = re.compile(r"[\u0400-\u04FF]+")  # Cyrillic block

def iter_texts(ds) -> Iterable[str]:
    for ex in ds:
        if "text" in ex:
            yield ex["text"] # for 'chitanka' dataset
        if "messages" in ex:
            for t in ex["messages"]:
                for c in t.get("content", []):
                    if c.get("type") == "text" and c.get("text"):
                        yield c["text"]
        elif "texts" in ex:
            for qa in ex["texts"]:
                if qa.get("user"):      yield qa["user"]
                if qa.get("assistant"): yield qa["assistant"]

def chunked(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf = []
    if buf: yield buf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="unsloth/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dataset", default="petkopetkov/chitanka") # petkopetkov/FineVision-bg
    ap.add_argument("--name", default="default") # a_okvqa
    ap.add_argument("--split", default="train")
    ap.add_argument("--min_freq", type=int, default=10)
    ap.add_argument("--max_pieces", type=int, default=2)   # add words split into > max_pieces
    ap.add_argument("--top_k", type=int, default=100000)
    ap.add_argument("--sample_ratio", type=float, default=1.0)  # e.g., 0.5 to speed up
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Loading tokenizer & dataset…")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=args.streaming)
    if not args.streaming and args.sample_ratio < 1.0:
        # quick downsample to speed things up
        ds = ds.shuffle(seed=3407).select(range(int(len(ds) * args.sample_ratio)))

    # 1) Count Cyrillic words
    logging.info("Counting Bulgarian word frequencies...")
    freq = Counter()
    total_rows = None if args.streaming else len(ds)
    for text in tqdm(iter_texts(ds), total=total_rows, desc="scan"):
        for w in CYR.findall(text.lower()):
            if 1 <= len(w) <= 40:
                freq[w] += 1

    # 2) Candidate words the current tokenizer over-splits
    logging.info("Selecting candidates the tokenizer splits too much…")
    vocab = set(tok.get_vocab().keys())
    # sort once; we will batch-check piece counts
    common = [(w, f) for w, f in freq.most_common() if f >= args.min_freq and w not in vocab]
    words = [w for w, _ in common]
    selected: List[str] = []

    for batch in tqdm(chunked(words, 1024), total=(len(words)+1023)//1024, desc="tokenize"):
        enc = tok(batch, add_special_tokens=False)
        for w, ids in zip(batch, enc["input_ids"]):
            if len(ids) > args.max_pieces:
                selected.append(w)
                if len(selected) >= args.top_k:
                    break
        if len(selected) >= args.top_k:
            break

    # 3) Add tokens & save
    logging.info(f"Adding {len(selected)} tokens to tokenizer…")
    added = tok.add_tokens(selected)
    os.makedirs(args.out_dir, exist_ok=True)
    tok.save_pretrained(args.out_dir)

    meta = {
        "base_model": args.base_model,
        "dataset": f"{args.dataset}:{args.name}/{args.split}",
        "min_freq": args.min_freq,
        "max_pieces": args.max_pieces,
        "top_k": args.top_k,
        "sample_ratio": args.sample_ratio,
        "num_candidates": len(selected),
        "num_added": added,
    }
    with open(os.path.join(args.out_dir, "bg_added_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(meta | {"tokens": selected[:added]}, f, ensure_ascii=False, indent=2)

    logging.info(f"[expand] added={added} saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
