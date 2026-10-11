import math


def block_flops(d, d_ff, top_k):
    # TODO: 4*d*d + top_k*3*d*d_ff
    # 4: q,k,v,o proj mats
    # 3: gate, up, down proj mats
    return 4*d*d + top_k*3*d*d_ff


def block_params(d, d_ff, n_experts):
    # TODO: 4*d*d + n_experts*3*d*d_ff
    return 4*d*d + n_experts*3*d*d_ff


def smelt_match(
    d, n_blocks, n_experts, top_k,
    ff_mult, kv_heads, head_dim,
    loop_fraction=0.5
):
    # compare looped version against the base model with **the same computational resources**!
    # constraint： flop do not change

    # 1. compute Base model costs
    d_ff = ff_mult * d
    base_flops = n_blocks * block_flops(d, d_ff, top_k)
    base_params = n_blocks * block_params(d, d_ff, n_experts)
    base_kv = n_blocks * 2 * kv_heads * head_dim

    # 2: create looped components
    lambd = 1 + loop_fraction
    applications = round(n_blocks * lambd)
    
    # n_block * d^2 = n_blocks * \lambda * d_new^2
    # -> d_new = d / \sqrt(\lambda)
    d_new = int(d / math.sqrt(lambd))   # ensure multiple of 8.
    d_new = (d_new // 8) * 8 
    d_ff_new = ff_mult * d_new

    # add experts with total block parameters reach the base model's
    experts_new = 1
    while n_blocks * block_params(d_new, d_ff_new, experts_new) < base_params:
        experts_new += 1
    # kv cache would accumulates with block increasing
    # proof: 2*(looped_N)*L*H_KV(new)*d=2*(looped_N)*L*H_KV(base)*d
    kv_heads_new = max(1, round(kv_heads / lambd))

    # 3. compute Looped model costs
    looped_flops = applications * block_flops(d_new, d_ff_new, top_k)
    # parameter is fixed, not growing with number of applications
    looped_params = n_blocks * block_params(d_new, d_ff_new, experts_new)
    looped_kv = applications * 2 * kv_heads_new * head_dim

    # Return cost ratios
    return {
        "d_new": d_new,
        "d_ff_new": d_ff_new,
        "experts_new": experts_new,
        "kv_heads_new": kv_heads_new,
        "applications": applications,
        "flops_ratio": round(looped_flops / base_flops, 4),
        "params_ratio": round(looped_params / base_params, 4),
        "kv_ratio": round(looped_kv / base_kv, 4),
    }
    



