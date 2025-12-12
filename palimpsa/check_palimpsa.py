import torch
import sys
import math
import triton
import time
from palimpsa.ops import chunk_palimpsa, fused_recurrent_palimpsa, palimpsa_ref

print("cuda available: ", torch.cuda.is_available())
torch.set_default_device("cuda")

# Use Float32 for rigorous debugging
dtype = torch.float32
device = "cuda"

def print_grad_stats(ref_grads, tri_grads):
    print("-" * 105)
    print(f"{'Name':<8} | {'Close':<5} | {'Max Diff':<12} | {'Ref Mean':<12} | {'Ref Std':<12} | {'Rel Err %':<12}")
    print("-" * 105)
    
    for name in ref_grads:
        g_ref = ref_grads[name]
        g_tri = tri_grads[name]
        
        if g_ref is None or g_tri is None:
            print(f"{name:<8} | {'NONE':<5}")
            continue
            
        diff = (g_ref - g_tri).abs().max().item()
        ref_mean = g_ref.abs().mean().item()
        ref_std = g_ref.std().item()
        
        rel_err = (diff / (ref_mean + 1e-9)) * 100

        if ref_mean > 1000:
            close = rel_err < 0.1 
        else:
            close = torch.allclose(g_ref, g_tri, atol=1e-2, rtol=1e-2)
            
        print(f"{name:<8} | {str(close):<5} | {diff:<12.6f} | {ref_mean:<12.4e} | {ref_std:<12.4e} | {rel_err:<12.4f}")
    print("-" * 105)

def check_equivalence_scan_bayes():
    torch.manual_seed(42)
    
    # Constants
    B, L, H, D, N = 4, 511, 4, 65, 31
    EPS = 1e-6 
    
    # -------------------------------------------------------------------------
    # DATA GENERATION
    # -------------------------------------------------------------------------
    v = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    q = torch.randn(B, L, H, N, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, N, device=device, dtype=dtype, requires_grad=True)

    b_d = torch.sigmoid(torch.randn(B, L, H, D, device=device, dtype=dtype))
    gt_d = torch.sigmoid(torch.randn(B, L, H, device=device, dtype=dtype))
    g_d   = torch.abs(torch.randn(H, device=device, dtype=dtype)) * 10 + 1
    Ip_d = torch.abs(torch.randn(H, device=device, dtype=dtype)) + 0.1

    b = b_d.clone().detach().requires_grad_(True)
    gt = gt_d.clone().detach().requires_grad_(True)
    g = g_d.clone().detach().requires_grad_(True)
    Ip = Ip_d.clone().detach().requires_grad_(True)

    # Clones for Triton Chunk
    v2 = v.clone().detach().requires_grad_(True)
    q2 = q.clone().detach().requires_grad_(True)
    k2 = k.clone().detach().requires_grad_(True)
    b2 = b.clone().detach().requires_grad_(True)
    gt2 = gt.clone().detach().requires_grad_(True)
    g2 = g.clone().detach().requires_grad_(True)
    Ip2 = Ip.clone().detach().requires_grad_(True)

    # Clones for Fused Recurrent
    v3 = v.clone().detach().requires_grad_(True)
    q3 = q.clone().detach().requires_grad_(True)
    k3 = k.clone().detach().requires_grad_(True)
    b3 = b.clone().detach().requires_grad_(True)
    gt3 = gt.clone().detach().requires_grad_(True)
    g3 = g.clone().detach().requires_grad_(True)
    Ip3 = Ip.clone().detach().requires_grad_(True)

    target = torch.randn(B, L, H, D, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # TEST 1: Standard Batch
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 1: Standard Batch (Fixed Length)")
    print("="*60)

    # 1. Ref
    (out_ref, out_ref_var), _, _ = palimpsa_ref(
        q, k, v, b, gt, g, Ip, scale=1, output_final_state=True, output_uncertainty=True
    )
    
    # 2. Chunk
    (out_chunk, out_chunk_var), _, _ = chunk_palimpsa(
        q2, k2, v2, b2, gt2, g2, Ip2, scale=1, output_final_state=True, output_uncertainty=True
    )

    # 3. Fused
    (out_fused, out_fused_var), _, _ = fused_recurrent_palimpsa(
        q3, k3, v3, b3, gt3, g3, Ip3, scale=1, output_final_state=True, output_uncertainty=True
    )

    diff_mean_chunk = (out_ref - out_chunk).abs().max().item()
    diff_var_chunk = (out_ref_var - out_chunk_var).abs().max().item()
    print(f"Mean Diff (Chunk): {diff_mean_chunk:.6f} | Var Diff: {diff_var_chunk:.6f}")

    loss_ref = ((out_ref - target)**2 / (out_ref_var + EPS)).sum()
    loss_chunk = ((out_chunk - target)**2 / (out_chunk_var + EPS)).sum()
    
    for t in [v, q, k, b, gt, g, Ip, v2, q2, k2, b2, gt2, g2, Ip2]:
        if t.grad is not None: t.grad.zero_()

    loss_ref.backward()
    loss_chunk.backward()

    grads_ref = {"dq": q.grad, "dk": k.grad, "dv": v.grad, "db": b.grad, "dgt": gt.grad, "dg": g.grad, "dIp": Ip.grad}
    grads_tri = {"dq": q2.grad, "dk": k2.grad, "dv": v2.grad, "db": b2.grad, "dgt": gt2.grad, "dg": g2.grad, "dIp": Ip2.grad}
    print_grad_stats(grads_ref, grads_tri)


    # -------------------------------------------------------------------------
    # TEST 2: Variable Sequence Length (cu_seqlens)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 2: Variable Sequence Length (cu_seqlens)")
    print("="*60)

    total_tokens = B * L
    seqlens = torch.tensor([128, 64, total_tokens - 128 - 64], dtype=torch.int32, device=device)
    cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32, device=device), seqlens.cumsum(0)])
    
    q_pack = q2.detach().reshape(1, total_tokens, H, N).requires_grad_(True)
    k_pack = k2.detach().reshape(1, total_tokens, H, N).requires_grad_(True)
    v_pack = v2.detach().reshape(1, total_tokens, H, D).requires_grad_(True)
    b_pack = b2.detach().reshape(1, total_tokens, H, D).requires_grad_(True)
    gt_pack = gt2.detach().reshape(1, total_tokens, H).requires_grad_(True)
    g_pack = g2.detach().clone().requires_grad_(True)
    Ip_pack = Ip2.detach().clone().requires_grad_(True)
    target_pack = target.reshape(1, total_tokens, H, D)

    # Chunk Packed
    (out_pack, out_pack_var), _, _ = chunk_palimpsa(
        q_pack, k_pack, v_pack, b_pack, gt_pack, g_pack, Ip_pack, 
        scale=1, output_final_state=True, output_uncertainty=True, cu_seqlens=cu_seqlens
    )

    # Ref Packed (Looping)
    ref_grads_pack = {k: torch.zeros_like(v) for k, v in [("dq",q_pack), ("dk",k_pack), ("dv",v_pack), ("db",b_pack), ("dgt",gt_pack), ("dg",g_pack), ("dIp",Ip_pack)]}
    
    start = 0
    max_diff_fwd = 0
    for i in range(len(seqlens)):
        end = start + seqlens[i].item()
        
        # Slices
        q_s, k_s, v_s, b_s, gt_s = [x[:, start:end].detach().requires_grad_(True) for x in [q_pack, k_pack, v_pack, b_pack, gt_pack]]
        g_s, Ip_s = g_pack.detach().requires_grad_(True), Ip_pack.detach().requires_grad_(True)
        t_s = target_pack[:, start:end]

        (o_s, o_var_s), _, _ = palimpsa_ref(q_s, k_s, v_s, b_s, gt_s, g_s, Ip_s, scale=1, output_final_state=True, output_uncertainty=True)
        
        max_diff_fwd = max(max_diff_fwd, (o_s - out_pack[:, start:end]).abs().max().item())
        
        loss_s = ((o_s - t_s)**2 / (o_var_s + EPS)).sum()
        loss_s.backward()
        
        ref_grads_pack["dq"][:, start:end] = q_s.grad
        ref_grads_pack["dk"][:, start:end] = k_s.grad
        ref_grads_pack["dv"][:, start:end] = v_s.grad
        ref_grads_pack["db"][:, start:end] = b_s.grad
        ref_grads_pack["dgt"][:, start:end] = gt_s.grad
        ref_grads_pack["dg"] += g_s.grad
        ref_grads_pack["dIp"] += Ip_s.grad
        start = end

    print(f"Packed Forward Mean Diff: {max_diff_fwd:.6f}")
    
    loss_pack = ((out_pack - target_pack)**2 / (out_pack_var + EPS)).sum()
    loss_pack.backward()
    
    pack_grads = {"dq": q_pack.grad, "dk": k_pack.grad, "dv": v_pack.grad, "db": b_pack.grad, "dgt": gt_pack.grad, "dg": g_pack.grad, "dIp": Ip_pack.grad}
    print_grad_stats(ref_grads_pack, pack_grads)


    # -------------------------------------------------------------------------
    # TEST 3: Initial States (State Passing)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("TEST 3: Initial States (Passing h_0 > Prior)")
    print("="*60)

    # 1. Generate Random Initial States
    # mu0 can be anything
    mu0 = torch.randn(B, H, D, N, device=device, dtype=dtype, requires_grad=True)
    
    # 2. I0 MUST be > Ip. 
    # Logic: Ip broadcasted + ReLU(Random) + 1.0 margin
    Ip_bc = Ip.detach().view(1, H, 1, 1).expand(B, H, D, N)
    I0_val = Ip_bc + torch.relu(torch.randn(B, H, D, N, device=device, dtype=dtype)) + 1.0
    I0 = I0_val.clone().detach().requires_grad_(True)

    # Clones for Triton
    mu0_tri = mu0.clone().detach().requires_grad_(True)
    I0_tri = I0.clone().detach().requires_grad_(True)

    # Ref Run with separate Initial State args
    (out_ref_s, out_ref_var_s), _, _ = palimpsa_ref(
        q, k, v, b, gt, g, Ip, 
        initial_mu_state=mu0, initial_I_state=I0,  # <--- SEPARATE ARGS
        scale=1, output_final_state=True, output_uncertainty=True,
    )

    # Chunk Run with separate Initial State args
    (out_chunk_s, out_chunk_var_s), _, _ = chunk_palimpsa(
        q2, k2, v2, b2, gt2, g2, Ip2, scale=1, 
        output_final_state=True, output_uncertainty=True,
        initial_mu_state=mu0_tri, initial_I_state=I0_tri # <--- SEPARATE ARGS
    )

    # Forward Check
    diff_mean_s = (out_ref_s - out_chunk_s).abs().max().item()
    diff_var_s = (out_ref_var_s - out_chunk_var_s).abs().max().item()
    print(f"With Initial State - Mean Diff: {diff_mean_s:.6f} | Var Diff: {diff_var_s:.6f}")

    # Backward Check (Gradients through Initial State)
    loss_ref_s = ((out_ref_s - target)**2 / (out_ref_var_s + EPS)).sum()
    loss_chunk_s = ((out_chunk_s - target)**2 / (out_chunk_var_s + EPS)).sum()

    loss_ref_s.backward()
    loss_chunk_s.backward()

    # We specifically want to check d_mu0 and d_I0
    grads_ref_s = {
        "dq": q.grad, "d_mu0": mu0.grad, "d_I0": I0.grad
    }
    grads_tri_s = {
        "dq": q2.grad, "d_mu0": mu0_tri.grad, "d_I0": I0_tri.grad
    }
    
    print("\nChecking Gradients flow to Initial States:")
    print_grad_stats(grads_ref_s, grads_tri_s)

if __name__ == "__main__":
    check_equivalence_scan_bayes()