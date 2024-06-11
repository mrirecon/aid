import torch
from tqdm import tqdm

def update_pbar_desc(pbar, metrics, labels):
    pbar_string = ''
    for metric, label in zip(metrics, labels):
        pbar_string += f'{label}: {metric:.7f}; '
    pbar.set_description(pbar_string)

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)

@torch.no_grad()
def ddnm_retrospective(model, x_shape, x0, betas, T, noise=None, mag=False):
    skip = 1000 / T
    n = x_shape[0]
    times = get_schedule_jump(T, 1, 1)
    time_pairs = list(zip(times[:-1], times[1:]))
    pbar = tqdm(time_pairs)

    if noise is not None:
        xt = noise
    else:
        xt = torch.randn(*x_shape, device=x0.device)

    for i, j in pbar:
        i, j = i*skip, j*skip
        if j<0: j=-1 

        t       = (torch.ones(n) * i).to(xt.device)
        next_t  = (torch.ones(n) * j).to(xt.device)
        at      = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())

        et = model.forward_normal(torch.concat([x0,xt],dim=-1), t)
        if not mag:
            et = et[:, :, :2]
        else:
            et = et[:, :, :1]

        xt = (1/at.sqrt()) * (xt - et * (1 - at).sqrt()) 
        xt = at_next.sqrt() * xt + torch.randn_like(xt) * (1 - at_next).sqrt()[0, 0, 0, 0]
    return xt

@torch.no_grad()
def ddnm_prospective(model, x_shape, x0, idx_t, betas, T, noise=None, cond_func=None, step_size=0.1, arg_iters=1, mag=False):
    skip = 1000 / T
    n = x_shape[0]
    times = get_schedule_jump(T, 1, 1)
    time_pairs = list(zip(times[:-1], times[1:]))
    pbar = tqdm(time_pairs)

    if noise is not None:
        xt = noise
    else:
        xt = torch.randn(*x_shape, device=x0.device)

    for i, j in pbar:
        i, j = i*skip, j*skip
        if j<0: j=-1 

        t       = (torch.ones(n) * i).to(x0.device)
        next_t  = (torch.ones(n) * j).to(x0.device)
        at      = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())

        et = model.forward_sample(torch.concat([x0, xt],dim=1), t, idx_t=idx_t)
        if not mag:
            et = et[:, :, :2]
        else:
            et = et[:, :, :1]

        xt = (1/at.sqrt()) * (xt - et * (1 - at).sqrt()) 

        if cond_func is not None:
            for _ in range(arg_iters): 
                xt = xt + cond_func(xt) * step_size

        xt = at_next.sqrt() * xt + torch.randn_like(xt) * (1 - at_next).sqrt()[0, 0, 0, 0] 
    return xt

def ddnm_prospective_loop(x0, model, x_shape, seq_length, betas, T, extra_steps=0, warm_start=True, noise=None, mag=False):

    i = 0
    assert seq_length > 0
    assert extra_steps >= 0

    if extra_steps > 0:
        holder = []

    while i < seq_length + extra_steps:
        if not warm_start and i < seq_length:
            idx_t = i
        else:
            idx_t = -1

        if i > 0:
            if not warm_start and i < seq_length:
                x0[:, i-1, ...] = new_elem
            else:
                x0 = torch.cat((x0[:, 1:, ...], new_elem), dim=1)

        new_elem = ddnm_prospective(model, x_shape, x0, idx_t, betas, T, noise, mag=mag)

        if i >= seq_length - 1:

            if extra_steps == 0:
                if warm_start:
                    return torch.cat ((x0[:, 1:, ...], new_elem), dim=1)
                else:
                    x0[:, i, ...] = new_elem
                    return x0
            else:
                if i == seq_length + extra_steps - 1:
                    x0 = torch.cat((x0[:, 1:, ...], new_elem), dim=1)
                    return torch.cat((torch.cat(holder, dim=1), x0), dim=1)
                # with extra_steps, we store the first element of x0 that will be dropped in the next iteration
                else:
                    holder.append(x0[:, 1:2, ...])
        i += 1





if False:
    def ddnm_sample_cond(xt, x0, idx_t, model, b, T, step_size, arg_iters, cond_func=None):
        skip = 1000 / T
        n = xt.size(0)

        times = get_schedule_jump(T, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))        

        pbar = tqdm(time_pairs)
        
        for i, j in pbar:
            i, j = i*skip, j*skip
            if j<0: j=-1 

            t       = (torch.ones(n) * i).to(xt.device)
            next_t  = (torch.ones(n) * j).to(xt.device)
            at      = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())

            et = model.forward_sample(torch.concat([x0,xt],dim=1), t, idx_t=idx_t)[:, :, :2]

            xt = (1/at.sqrt()) * (xt - et * (1 - at).sqrt()) 
            
            if cond_func is not None:
                for _ in range(arg_iters): 
                    xt = xt + cond_func(xt) * step_size
            xt = at_next.sqrt() * xt + torch.randn_like(xt) * (1 - at_next).sqrt()[0, 0, 0, 0] 

        return xt

    def ddnm_diffusion(xt, prompt, idx_t, model, b, T, sigma_y, step_size, arg_iters,  cond_func=None):
        skip = 1000 / T
        n = xt.size(0)
        losses = []

        times = get_schedule_jump(T, 1, 1)
        time_pairs = list(zip(times[:-1], times[1:]))        

        pbar = tqdm(time_pairs)
        pbar_labels = ['loss', 'mean', 'min', 'max']

        k_init = 0
        b_init = 0
        
        # Reverse diffusion + Nila-DC
        for i, j in pbar:
            i, j = i*skip, j*skip
            if j<0: j=-1 

            t       = (torch.ones(n) * i).to(xt.device)
            next_t  = (torch.ones(n) * j).to(xt.device)
            at      = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            sigma_t = (1 - at_next).sqrt()[0, 0, 0, 0]
            a       = at_next.sqrt()[0, 0, 0, 0]
            
            et = model.forward_sample(torch.concat([prompt,xt],dim=1), t, idx_t=idx_t)[:, :, :2]

            xt = (1/at.sqrt()) * (xt - et * (1 - at).sqrt()) # Eq.6
            
            for _ in range(arg_iters): # Fig.2 (a) (for best DC)
                meas_grad = cond_func(xt) 

                if sigma_t / a >  sigma_y: # Eq.10 (lambda function)
                    factor = 1
                else:
                    if k_init == 0 and b_init==0:
                        k_init = 0.2 / (-1 * i)
                        b_init = -999 * k_init
                    factor = k_init * (999 - i) + b_init
                
                xt = xt + factor * meas_grad * step_size

            xt_1 = at_next.sqrt() * xt + torch.randn_like(xt) * sigma_t # Eq.11

            metrics = [(meas_grad).norm(), (xt).abs().mean(), (xt).abs().min(), (xt).abs().max()]
            update_pbar_desc(pbar, metrics, pbar_labels)
            xt = xt_1
        return xt