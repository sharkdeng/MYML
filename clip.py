
def _clip(grads, max_norm):
    '''
    [-max_norm/2, max_norm/2]
    '''
    total_norm = 0
    for g in grads:
        param_norm = g.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)
    return clip_coef
