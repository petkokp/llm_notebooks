from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os
from gemma.paligemma_for_conditional_generation import PaliGemmaForConditionalGeneration
from gemma.config.paligemma_config import PaliGemmaConfig

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # find the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # load safetensors files in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # load model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    model = PaliGemmaForConditionalGeneration(config).to(device)
    model.load_state_dict(tensors, strict=False)
    model.tie_weights()

    return (model, tokenizer)