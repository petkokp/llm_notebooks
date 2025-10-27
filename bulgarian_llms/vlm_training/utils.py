import math
from torch.utils.data import Dataset

class UnslothView(Dataset):
    """Wrap a HF dataset row and expose a {messages: ...} dict for Unsloth."""
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        ex = self.base[i]
        img = ex["images"][0]  # PIL.Image (from your dataset)
        msgs, first = [], True
        for qa in ex["texts"]:
            content = [{"type": "text", "text": qa["user"]}]
            if first:
                content.append({"type": "image", "image": img})  # attach once
                first = False
            msgs.append({"role": "user", "content": content})
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": qa["assistant"]}]})
        return {"messages": msgs}

def dynamic_steps(
    n_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    target_logs_per_epoch: int = 20,
    target_evals_per_epoch: int = 3,
    min_log_step: int = 1,
    min_eval_step: int = 10,
):
    dataloader_steps_per_epoch = math.ceil(n_examples / per_device_train_batch_size)
    update_steps_per_epoch = math.ceil(dataloader_steps_per_epoch / gradient_accumulation_steps)
    total_update_steps = update_steps_per_epoch * num_train_epochs

    logging_steps = max(min_log_step, max(1, update_steps_per_epoch // target_logs_per_epoch))
    eval_steps    = max(min_eval_step, max(1, update_steps_per_epoch // target_evals_per_epoch))
    save_steps    = eval_steps                                # save whenever we eval
    warmup_steps  = max(1, int(0.03 * total_update_steps))    # 3% warmup (tweak if you like)

    return {
        "steps_per_epoch": update_steps_per_epoch,
        "total_update_steps": total_update_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "warmup_steps": warmup_steps,
    }