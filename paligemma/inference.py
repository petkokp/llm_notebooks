from PIL import Image
import torch
import fire
from processing.processor import PaliGemmaProcessor
from gemma.model.kv_cache import KVCache
from gemma.paligemma_for_conditional_generation import PaliGemmaForConditionalGeneration
from load_hf_model import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs