"""Randomized Hadamard incoherence processing (adapted from QUIP-Sharp
lib/algo/quip.py).

The pure-torch Hadamard transform core lives in
neural_network_compression/bqqkernel/hadamard.py (single source of truth, also
used by the incoherent inference module IncoherentBinaryQuadratic). This module
only adds the RHT_H / RHT_W / incoherence_process wrappers on top of it.
"""
import os
import sys

try:
    from neural_network_compression.bqqkernel.hadamard import (  # noqa: F401
        matmul_hadU, matmul_hadUt, get_hadK, is_pow2,
    )
except ImportError:
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from neural_network_compression.bqqkernel.hadamard import (  # noqa: F401
        matmul_hadU, matmul_hadUt, get_hadK, is_pow2,
    )


# ---------------------------------------------------------------------------
# Randomized Hadamard Transform incoherence (from QUIP-Sharp lib/algo/quip.py).
# SU: in-side sign vector (shared with H); SV: out-side sign vector.
# ---------------------------------------------------------------------------

def RHT_H(H, SU):
    """Incoherence-transform the Hessian: H~ = V H V^T with V = Hadamard @ diag(SU)."""
    return matmul_hadUt(matmul_hadUt(H * SU).T * SU)


def RHT_W(W, SU, SV):
    """Incoherence-transform the weight W[out,in]: W~ = U W V^T (SV out-side, SU in-side)."""
    return matmul_hadUt(matmul_hadUt(W.T * SV).T * SU)


def incoherence_process(hatWr, SU, SV):
    """Invert the RHT on a (quantized) weight in the transformed space -> original space."""
    device = hatWr.device
    return (matmul_hadU((matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
