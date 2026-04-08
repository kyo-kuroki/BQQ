# Binary Quadratic Quantization (BQQ)

Binary Quadratic Quantization represents each weight matrix as a sum of **binary outer-product terms**:

$$W \approx \sum_{b=1}^{B} \bigl( a_b^{(0)}\, y_b z_b^\top + a_b^{(1)}\, y_b \mathbf{1}^\top + a_b^{(2)}\, \mathbf{1} z_b^\top + a_b^{(3)} \bigr)$$

where $y_b, z_b \in \{0,1\}^n$ and the scalar coefficients $a_b$ are jointly optimised by simulated annealing.
Each bit layer $b$ minimises the residual left by the previous layers, so **bit depth can be extended incrementally** without re-running earlier layers.

## Repository structure

```
BQQ/
├── quantizer.py              # Core BQQ algorithms
├── matrix_compression/       # Standalone matrix compression experiments
└── nn_compression/           # Neural-network quantization
    ├── lm/                   # Language model quantization (Qwen3.5, etc.)
    └── cv/                   # Vision model quantization (DeiT, etc.)
```

### `quantizer.py`

| Class | Description |
|-------|-------------|
| `BinaryQuadraticQuantization` | Original multi-bit implementation; includes an activation-aware variant |
| `BinaryQuadraticQuantization2` | Refactored class used by all current LM and CV workflows |

`BQQ2.bqq_large_matrix_multi_worker` tiles a weight matrix into patches and quantizes them in parallel via `multiprocessing`.

### `matrix_compression/`

Studies BQQ as a pure matrix compression method, evaluating the compression–reconstruction-error trade-off on synthetic and real-valued data, independent of any downstream task.

### `nn_compression/`

Applies BQQ to pretrained neural networks.
See [`nn_compression/README.md`](nn_compression/README.md) for full workflow documentation.

- **`lm/`** — Language model quantization with TSUBAME4 SGE array-job support.
- **`cv/`** — Vision model quantization.
