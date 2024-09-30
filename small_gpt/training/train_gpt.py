import os
import time
from contextlib import nullcontext
import torch
from models.gpt.gpt_model import GPT
from .get_batch import get_batch
from .learning_rate_decay_scheduler import get_learning_rate
from .estimate_loss import estimate_loss

def train_gpt(vocabulary_size, n_embeddings, n_heads, context_length, n_layers, dropout, bias, batch_size, train_data, val_data):
    # TODO MAKE THESE PARAMS ARGUMENTS
    out_dir = 'out'
    eval_interval = 100
    log_interval = 1
    eval_iters = 200
    eval_only = False # if True, script exits right after the first eval
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume'
    
    gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes

    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 10000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iterations = 500 # how many steps to warm up for
    learning_rate_decay_iterations = 10000 # should be ~= max_iters per Chinchilla
    min_learning_rate = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster
        
    seed_offset = 0
    tokens_per_iter = gradient_accumulation_steps * batch_size * context_length
    print(f"Tokens per iteration: {tokens_per_iter:,}")
        
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
    iter_num = 0
    best_val_loss = 1e9

    model_args = dict(n_layers=n_layers, n_heads=n_heads, n_embeddings=n_embeddings, context_length=context_length,
                    bias=bias, vocabulary_size=None, dropout=dropout)
    
    model_args_list = ['n_layers', 'n_heads', 'n_embeddings', 'context_length', 'bias', 'vocabulary_size']
    
    if init_from == 'scratch':
        model = GPT(vocabulary_size, n_embeddings, n_heads, context_length, n_layers, dropout, bias)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")

        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        
        for k in model_args_list:
            model_args[k] = checkpoint_model_args[k]

        model = GPT(vocabulary_size, n_embeddings, n_heads, context_length, n_layers, dropout, bias)
        state_dict = checkpoint['model']
    
        # fix state dictionary keys
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    
    if context_length < model.context_length:
        model.crop_context_length(context_length)
        model_args['context_length'] = context_length # ensure checkpoint has correct value
    
    model.to(device)
    
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory
    
    if compile:
        print("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == 'float16'))
    
    # training loop
    X, Y = get_batch('train', batch_size, context_length, train_data, val_data, device) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    #running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_learning_rate(iter_num, warmup_iterations, learning_rate, learning_rate_decay_iterations, min_learning_rate) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, batch_size, context_length, train_data, val_data, device)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"Saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for _ in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train', batch_size, context_length, train_data, val_data, device)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            # if local_iter_num >= 5: # let the training loop settle a bit
                # mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                # running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms") # mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break
