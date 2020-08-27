# Gradient clipping
def grad_clip(grads, max_norm):
    '''
    function: control the grads.norm(2) not succeed max_norm
    Args
        grads: list of tensor or a tensor
    '''
    # get norm(2)
    # v1
    # total_nrom = grads.norm(@)
  
    # v2
    # total_norm = 0
    # for g in grads:
    #     total_norm += grad.pow(2).sum()
    # total_norm = total_norm.sqrt()
    
    # v3
    total_norm = 0
    for g in grads:
        param_norm = g.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5


    # control
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)
    return clip_coef



# Gradient clipping
def grad_clip_v2(model, max_norm):
    '''
    result is same as grad_clip, but here requires model
    '''
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)




def clamp_params(model):
    '''
    control weights
    usage of register_hook
    '''
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        
        
        
       
# Gradient accumulation
def grad_accumulate(grads, accumation_step):
    '''
    This code for demonstration only, for practical use, you need to adjust variables
    Source: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
    '''
    loss = loss_fn(output, target)
    loss = loss / accumation_step

    # Accumulates grads
    loss.backward()

    if (i + 1) % accumation_step == 0:

        optimizer.step() # update the weight
        optimizer.zero_grad() # clear the grads
    
    
    
 
# Gradient penalty
def grad_penalty(grads):
    '''
    This code for demonstration only, for practical use, you need to adjust variables
    Source: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
    '''
    loss = loss_fn(output, target)

    # Creates gradients
    grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Get norm(2)
    # You can also try norm(1) for sparcity solution?
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()
    
    # Add the penalty
    loss = loss + grad_norm

    loss.backward()
