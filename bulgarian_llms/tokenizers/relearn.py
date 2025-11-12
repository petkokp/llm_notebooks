#!/usr/bin/env python

# python tokenizers/relearn.py --out_dir tokenizers/Qwen3-VL-2B-Instruct_relearn

from unsloth import FastVisionModel
import argparse, os, logging
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import iter_texts

os.environ["TOKENIZERS_PARALLELISM"] = "1"

def batch_concat_texts(ds, chars_target=4_000_000, streaming=False):
    buf, size = [], 0
    total = None if streaming else len(ds)
    for s in tqdm(iter_texts(ds), total=total, desc="gather"):
        if not s: continue
        buf.append(s)
        size += len(s)
        if size >= chars_target:
            yield "\n".join(buf); buf, size = [], 0
    if buf: yield "\n".join(buf)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="unsloth/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dataset", default="petkopetkov/chitanka") # petkopetkov/FineVision-bg
    ap.add_argument("--name", default="default") # a_okvqa
    ap.add_argument("--split", default="train")
    ap.add_argument("--vocab_size", type=int, default=0)
    ap.add_argument("--chars_per_batch", type=int, default=4_000_000)
    ap.add_argument("--sample_ratio", type=float, default=1.0)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Loading tokenizer & dataset…")
    
    base_tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    
    ds = load_dataset(args.dataset, name=args.name, split=args.split, streaming=args.streaming)
    if not args.streaming and args.sample_ratio < 1.0:
        ds = ds.shuffle(seed=3407).select(range(int(len(ds) * args.sample_ratio)))

    logging.info(f"Training new tokenizer from iterator...")

    # train_new_from_iterator preserves specials, we feed big text batches for speed
    new_tok = base_tok.train_new_from_iterator(
        batch_concat_texts(ds, args.chars_per_batch, args.streaming),
        vocab_size=len(base_tok),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    new_tok.save_pretrained(args.out_dir)
    # base_processor.image_processor.save_pretrained(args.out_dir)
    logging.info(f"[relearn-merges] vocab={len(new_tok)} saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
