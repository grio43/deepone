def install_flash_stub() -> None:
    """
    Provide a dummy `flash_attn_func` so the codebase can run on
    vanilla PyTorch builds without FlashAttention.
    """
    import builtins

    def _noop(*_args, **_kwargs):
        return None

    builtins.flash_attn_func = _noop