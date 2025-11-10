import torch
import tqdm
from typing import Callable, List, Tuple
import copy
from torch.func import hessian, vmap
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import itertools
import contextlib



def squared_norm_and_diag_hessians(f: Callable, arg_shapes: List[torch.Size], m: int = 128, device='cuda', generate_function=None):
    inputs = [torch.zeros(shape, requires_grad=True, device=device) for shape in arg_shapes]
    split_sizes = [x.numel() for x in inputs]

    # Forward and 1st-order gradients
    output = f(*inputs)
    if output.numel() != 1:
        raise ValueError("f must return a scalar output.")
    grads = torch.autograd.grad(output, inputs, create_graph=True)

    diag_hessians = []

    for i, x in enumerate(inputs):
        flat_x = x.view(-1)
        n = flat_x.numel()
        diag = torch.empty(n, device=x.device)

        # scalar function: partial over x (the i-th input)
        def scalar_fn(x_partial):
            x_reconstructed = x_partial.view_as(x)
            new_inputs = list(inputs)
            new_inputs[i] = x_reconstructed
            return f(*new_inputs)

        # compute Hessian diagonals m elements at a time
        for j in range(0, n, m):
            idx = slice(j, min(j + m, n))
            x_chunk = flat_x[idx].detach().requires_grad_(True)

            def chunk_fn(chunk):
                x_new = flat_x.clone()
                x_new[idx] = chunk
                return scalar_fn(x_new)

            hess_chunk = hessian(chunk_fn)(x_chunk)  # (m, m)
            diag_chunk = hess_chunk.diagonal(dim1=0, dim2=1)
            diag[idx] = diag_chunk.detach()

        diag_hessians.append(diag.view_as(x))

    # compute Hessian
    try:
        v = torch.cat([(v).reshape(-1) for v in inputs]) 

        def flat_f(v_flat):
            split = torch.split(v_flat, split_sizes)
            reshaped = [s.view(shape) for s, shape in zip(split, arg_shapes)]
            return f(*reshaped)

        Q_squared_norm = ((torch.func.hessian(flat_f)(v).detach())**2).sum()
        h_squared_norm = sum(((grad + 0.5*diag_hessian)**2).sum() - (diag_hessian**2).sum() for grad, diag_hessian in zip(grads, diag_hessians))
    except Exception as e:
        try:
            h, Q = generate_function(device='cpu')
            h_squared_norm, Q_squared_norm = (h**2).sum().to(device), (Q**2).sum().to(device)
        except Exception as e:
            print(f"Direct computing Hessian was failed: {e}. Computing row-wise.")
            h = torch.cat([g.reshape(-1) for g in grads])
            def row_hesse(h_i):
                grad_i = torch.autograd.grad(h_i, inputs, retain_graph=True)
                return (torch.cat([g.reshape(-1) for g in grad_i])**2).sum().detach().requires_grad_(False)
            # Hessianの各行を1個ずつ計算（メモリ効率よい）
            Q_squared_norm = 0
            for i in range(h.numel()):
                Q_squared_norm += row_hesse(h[i])
            h_squared_norm = sum(((grad + 0.5*diag_hessian)**2).sum() - (diag_hessian**2).sum() for grad, diag_hessian in zip(grads, diag_hessians))

    return h_squared_norm + Q_squared_norm, diag_hessians


def auto_grid_amfd(
    f: Callable,
    shapes: List[torch.Size],
    eta_vals: List[float],
    zeta_vals: List[float],
    num_rep: int,
    t_st=0.4,
    t_en=0.001,
    Nstep=None,
    squared_norm=None,
    diag_hessians=None,
    device='cuda:0',
    show_progress=True
):
    """
    Performs batched AMFD optimization over eta, zeta grid with repetitions.

    Args:
        f: Callable returning scalar value given list of tensors.
        shapes: List of shapes for input tensors.
        eta_vals: List of eta values to grid search.
        zeta_vals: List of zeta values to grid search.
        num_rep: Number of repetitions per eta/zeta pair.
        t_st, t_en: Temperature schedule.
        Nstep: Number of iterations.
        device: CUDA device.

    Returns:
        - List of binary solutions (batched).
        - Corresponding scalar function values.
    """
    torch._dynamo.reset()
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()

    grid = list(itertools.product(eta_vals, zeta_vals))
    B = len(grid) * num_rep

    # Shapes of inputs
    num_tensors = len(shapes)
    in_dims = (0,) * num_tensors
    sizes = [torch.prod(torch.tensor(s)).item() for s in shapes]
    total_vars = sum(sizes)

    if Nstep is None:
        Nstep = max(2000, total_vars)

    # Initialize variables
    before = [torch.rand((B, *shape), device=device) for shape in shapes]
    eta_tensor = torch.tensor([e for (e, _) in grid for _ in range(num_rep)], device=device).float()
    zeta_tensor = torch.tensor([z for (_, z) in grid for _ in range(num_rep)], device=device).float()

    after = [b + eta_tensor.view(B, *[1]*len(b.shape[1:])) * (0.5 - b) for b in before]

    delta_t = torch.tensor((t_st - t_en) / (Nstep - 1), device=device)
    now_temp = torch.tensor(t_st, device=device).float()
    eta = eta_tensor
    zeta = zeta_tensor

    # Compute coeffs (outside batch)
    if squared_norm is None or diag_hessians is None:
        squared_norm, diag_hessians = squared_norm_and_diag_hessians(f, shapes, device=device)
    coeff = torch.sqrt(total_vars / squared_norm)

    # Gradient function
    df = vmap(torch.func.grad(f, argnums=tuple(range(num_tensors))), in_dims=in_dims)
    @torch.compile(mode='reduce-overhead')
    def iteration_step(x_b_list, x_a_list):
        x_f_list = [(xa + zeta.view(B, *[1]*len(xa.shape[1:])) * (xa - xb)).detach()
                    for xb, xa in zip(x_b_list, x_a_list)]
        
        grads = df(*x_f_list)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        x_next_list = []
        
        for xb, xa, xf, g, dh in zip(x_b_list, x_a_list, x_f_list, grads, diag_hessians):
            mask = (xa == 0) | (xa == 1)
            # diag_hessiansの次元合わせ
            grad = (g + dh * (0.5 - xf)).masked_fill(mask, 0.0)

            update = 2 * xa - xb - eta.view(B, *[1]*len(xa.shape[1:])) * (
                coeff * grad + now_temp * (xa - 0.5))
            x_next = torch.clamp(update, 0.0, 1.0).detach()
            x_next_list.append(x_next)
        return x_a_list, x_next_list

    for _ in tqdm.trange(Nstep, mininterval=5.0, disable=not show_progress):
        before, after = iteration_step(before, after)
        before = [x.detach().clone() for x in before]
        after = [x.detach().clone() for x in after]
        now_temp -= delta_t

    # Final thresholding
    rounded = [torch.where(x > 0.5, 1.0, 0.0) for x in after]
    results = vmap(f, in_dims=in_dims)(*rounded)

    return rounded, results, eta_tensor, zeta_tensor







class ABQQ:
    def __init__(self):
        pass

    def qubo_squared_norm(self, l):
        """
        QUBO行列を明示せず、要素の二乗和だけをテンソル演算で計算。
        
        activation: (k, m)
        input_:     (k, n)
        l: int — yの第2軸 or zの第1軸サイズ
        
        戻り値: QUBO行列の要素の二乗和（x_i^2 = x_i を考慮）
        """
        # inner: (m, n) ← [i,j]×[i,s]
        inner = self.activation.T @ self.input  # shape: (m, n)

        # Q_{(y_{j,p}, z_{p,s})} = 4 * inner[j, s]
        # p 方向に l 回ずつ繰り返される → 重複考慮
        squared_sum = (inner ** 2).sum() * (16 * l)

        return squared_sum
    
    def diag_hessians(f: Callable, arg_shapes: List[torch.Size], m: int = 128, device='cuda'):
        inputs = [torch.zeros(shape, requires_grad=True, device=device) for shape in arg_shapes]

        # Forward and 1st-order gradients
        output = f(*inputs)
        if output.numel() != 1:
            raise ValueError("f must return a scalar output.")

        diag_hessians = []

        for i, x in enumerate(inputs):
            flat_x = x.view(-1)
            n = flat_x.numel()
            diag = torch.empty(n, device=x.device)

            # scalar function: partial over x (the i-th input)
            def scalar_fn(x_partial):
                x_reconstructed = x_partial.view_as(x)
                new_inputs = list(inputs)
                new_inputs[i] = x_reconstructed
                return f(*new_inputs)

            # compute Hessian diagonals m elements at a time
            for j in range(0, n, m):
                idx = slice(j, min(j + m, n))
                x_chunk = flat_x[idx].detach().requires_grad_(True)

                def chunk_fn(chunk):
                    x_new = flat_x.clone()
                    x_new[idx] = chunk
                    return scalar_fn(x_new)

                hess_chunk = hessian(chunk_fn)(x_chunk)  # (m, m)
                diag_chunk = hess_chunk.diagonal(dim1=0, dim2=1)
                diag[idx] = diag_chunk.detach()

            diag_hessians.append(diag.view_as(x))

        return  diag_hessians
    
    def objective_function(self, y, z):
        return (self.activation * (self.input@((2*y - 1)@(2*z - 1)).T)).sum() / torch.norm(self.activation)

    def run_bqq(self, weight, input, rank_scale, zetas, etas, Tinit, Tfin, Nstep, num_rep=1, seed=0, device='cuda:0'):
        torch.manual_seed(seed)
        torch.set_float32_matmul_precision('high')
        self.weight = weight
        self.input = input
        self.rank_scale = rank_scale
        self.activation = input @ weight.T
        m, n = weight.shape
        l = int(round(rank_scale * (m * n / (m + n))))
        squared_norm = self.qubo_squared_norm(l)
        diag_hessians = self.diag_hessians(self.objective_function, [(m, l), (l, n)], device=device)


        auto_grid_amfd(
            f=self.objective_function,
            shapes=[(m, l), (l, n)],
            eta_vals=etas,
            zeta_vals=zetas,
            num_rep=num_rep,
            t_st=Tinit,
            t_en=Tfin,
            Nstep=Nstep,
            squared_norm=squared_norm,
            diag_hessians=diag_hessians,
            device=device
        )

        return self.weight 