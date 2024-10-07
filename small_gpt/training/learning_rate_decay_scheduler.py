import math

# cosine with warmup
def get_learning_rate(it, warmup_iterations, learning_rate, learning_rate_decay_iterations, min_learning_rate):
    # linear warmup for warmup_iterations steps
    if it < warmup_iterations:
        return learning_rate * it / warmup_iterations
    # if it > lr_decay_iterations, return min learning rate
    if it > learning_rate_decay_iterations:
        return min_learning_rate
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iterations) / (learning_rate_decay_iterations - warmup_iterations)
    assert 0 <= decay_ratio <= 1
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # from 0 to 1
    return min_learning_rate + coefficient * (learning_rate - min_learning_rate)