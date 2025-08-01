+"""
+Provide a dummy `flash_attn_func` so that model code written for
+Flash‑Attention v2 can import it even when the C++/CUDA extension is absent.
+"""

def install_flash_stub() -> None:
    import builtins

    def _noop(*_args, **_kwargs):
        return None

    builtins.flash_attn_func = _noop