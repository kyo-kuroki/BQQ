"""Unpickle shims for checkpoints saved against older dependency versions.

`torch.load(..., pickle_module=dill)` on the BQQ model checkpoints resolves
references into `transformers` internals.  transformers 5.5.3 removed
`transformers.core_model_loading.PrefixChange`, which older checkpoints (e.g.
quantized_models/Qwen3.5-2B-*.pth) still name, so unpickling dies with

    AttributeError: Can't get attribute 'PrefixChange' on
    <module 'transformers.core_model_loading'>

PrefixChange is part of the weight-key renaming machinery used at *load* time,
not at inference, so a placeholder is enough to get the object graph back.

Import this module before torch.load:

    import tools.env.pickle_compat  # noqa: F401
"""

from __future__ import annotations


def _install_prefixchange_stub() -> None:
    try:
        import transformers.core_model_loading as cml
    except Exception:
        return
    if hasattr(cml, "PrefixChange"):
        return

    class PrefixChange:  # pragma: no cover - only ever reconstructed by pickle
        """Placeholder for the class transformers 5.5.3 deleted."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    cml.PrefixChange = PrefixChange


_install_prefixchange_stub()
