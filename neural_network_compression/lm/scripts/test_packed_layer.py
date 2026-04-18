"""
Unit test: verify PackedBinaryQuadratic produces identical output to BinaryQuadratic.

Tests with random data matching the actual gs=64 / rank=32 / bit_width=2 shapes,
then also loads one real layer from the existing packed model file.
"""
import sys, os, math
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import bqq_modules  # noqa: F401
from bqq_modules import BinaryQuadratic, PackedBinaryQuadratic

torch.manual_seed(0)


def make_random_bq(bit_width=2, row_width=4, col_width=4,
                    y_row=64, inter_dim=32, z_col=64):
    Y = (torch.rand(bit_width, row_width, col_width, y_row, inter_dim) > 0.5)
    Z = (torch.rand(bit_width, row_width, col_width, inter_dim, z_col) > 0.5)
    A = torch.randn(bit_width, row_width, col_width, 4)
    return BinaryQuadratic(Y.float(), Z.float(), A)


def test_layer_equivalence(bit_width=2, row_width=4, col_width=4,
                            y_row=64, inter_dim=32, z_col=64):
    print(f"\n--- Random layer test (bw={bit_width} r={row_width} c={col_width} "
          f"y_row={y_row} inter={inter_dim} z_col={z_col}) ---")

    bq = make_random_bq(bit_width, row_width, col_width, y_row, inter_dim, z_col)
    pbq = PackedBinaryQuadratic.from_unpacked(bq)

    # Check that packed sums match original sums
    Y_sum_orig = bq.Y.sum(dim=-1, keepdim=True).float()
    Z_sum_orig = bq.Z.sum(dim=-2, keepdim=True).float()
    Y_sum_diff = (Y_sum_orig - pbq.Y_sum_i16.float()).abs().max().item()
    Z_sum_diff = (Z_sum_orig - pbq.Z_sum_i16.float()).abs().max().item()
    print(f"  Y_sum match: max|diff|={Y_sum_diff:.0f}  {'OK' if Y_sum_diff==0 else 'FAIL'}")
    print(f"  Z_sum match: max|diff|={Z_sum_diff:.0f}  {'OK' if Z_sum_diff==0 else 'FAIL'}")

    # Check weight reconstruction
    W_orig = bq.get_weight(dtype=torch.float32)
    W_packed = pbq.get_weight(dtype=torch.float32)
    w_diff = (W_orig - W_packed).abs().max().item()
    print(f"  Weight max|diff|={w_diff:.2e}  {'OK' if w_diff < 1e-5 else 'FAIL'}")

    # Check forward pass (bfloat16, matching inference dtype)
    in_features = col_width * z_col
    X = torch.randn(2, 8, in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        out_orig = bq.forward(X)
        out_pack = pbq.forward(X)
    fwd_diff = (out_orig.float() - out_pack.float()).abs().max().item()
    print(f"  Forward max|diff|={fwd_diff:.2e}  {'OK' if fwd_diff < 1e-3 else 'FAIL'}")

    return w_diff < 1e-5 and fwd_diff < 1e-3


def test_packed_model_first_layer(packed_path):
    """Load just the first transformer block from the packed model and test one layer."""
    print(f"\n--- Real layer from packed model ---")
    print(f"  Loading packed model (map_location=cpu) ...")
    model = torch.load(packed_path, map_location="cpu", weights_only=False)

    # Find first PackedBinaryQuadratic
    pbq = None
    name_found = None
    for name, mod in model.named_modules():
        if isinstance(mod, PackedBinaryQuadratic):
            pbq = mod
            name_found = name
            break

    if pbq is None:
        print("  No PackedBinaryQuadratic found in model!")
        return False

    print(f"  Found {name_found}: bit_width={pbq.bit_width} row={pbq.row_width} "
          f"col={pbq.col_width} y_row={pbq.y_row} inter={pbq.inter_dimension} z_col={pbq.z_col}")

    # Verify packed matmul vs float matmul
    # Reconstruct Y, Z from packed form using numpy unpackbits
    k8 = pbq._k8
    inter = pbq.inter_dimension

    Y_flat_np = np.unpackbits(pbq.Y_packed.numpy().reshape(-1, k8), axis=-1
                               )[:, :inter]
    Z_flat_np = np.unpackbits(pbq.Z_packed.numpy().reshape(-1, k8), axis=-1
                               )[:, :inter]

    Y_shape = (pbq.bit_width, pbq.row_width, pbq.col_width, pbq.y_row, inter)
    Z_shape = (pbq.bit_width, pbq.row_width, pbq.col_width, pbq.z_col, inter)
    Y_bool = torch.from_numpy(Y_flat_np.reshape(Y_shape).astype(np.uint8)).bool()
    Z_bool = torch.from_numpy(Z_flat_np.reshape(Z_shape).astype(np.uint8)).bool()
    # Z was stored as [z_col, inter], need to transpose back to [inter, z_col]
    Z_bool = Z_bool.permute(0, 1, 2, 4, 3)

    # Reconstruct W via float matmul
    dtype = torch.float32
    W_core_float = torch.matmul(Y_bool.float(), Z_bool.float())
    Y_sum_f = Y_bool.float().sum(-1, keepdim=True)
    Z_sum_f = Z_bool.float().sum(-2, keepdim=True)
    W_ref = (pbq.a.float() * W_core_float
             + pbq.b.float() * Y_sum_f
             + pbq.c.float() * Z_sum_f)
    W_ref = W_ref.sum(0) + pbq.d.float()
    W_ref = W_ref.permute(0,2,1,3).reshape(pbq.row_width*pbq.y_row, pbq.col_width*pbq.z_col)

    W_packed = pbq.get_weight(dtype=torch.float32)
    w_diff = (W_ref - W_packed).abs().max().item()
    print(f"  W_ref vs packed weight max|diff|={w_diff:.2e}  {'OK' if w_diff < 1e-4 else 'FAIL'}")

    del model
    return w_diff < 1e-4


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=str, default=None,
                        help="Path to packed model .pth for real-layer test")
    args = parser.parse_args()

    results = []

    # Test with various shapes
    results.append(test_layer_equivalence(bit_width=2, row_width=4, col_width=4,
                                           y_row=64, inter_dim=32, z_col=64))
    results.append(test_layer_equivalence(bit_width=3, row_width=2, col_width=8,
                                           y_row=32, inter_dim=16, z_col=32))
    # Edge case: inter_dim not multiple of 8 (e.g., rank=12 for small gs=32 with odd scaling)
    results.append(test_layer_equivalence(bit_width=2, row_width=2, col_width=2,
                                           y_row=32, inter_dim=12, z_col=32))

    if args.packed:
        results.append(test_packed_model_first_layer(args.packed))

    print(f"\n{'='*40}")
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*40}")
    sys.exit(0 if all(results) else 1)
