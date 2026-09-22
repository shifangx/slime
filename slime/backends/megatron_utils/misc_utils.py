def strip_param_name_prefix(name: str):
    prefix = "module."
    while name.startswith(prefix):
        name = name.removeprefix(prefix)
    return name


def has_gated_linear_unit(args) -> bool:
    """Whether ``linear_fc1`` holds the fused ``cat(gate, up)`` rather than one projection.

    There is no ``--gated-linear-unit`` flag to read: megatron derives
    ``config.gated_linear_unit`` from the activation in
    ``training/argument_utils.py``, and only ``--swiglu`` and ``--quick-geglu``
    set it. Models such as nemotron_h are ``--squared-relu`` and ungated -- their
    ``linear_fc1`` is ``up_proj`` alone, half the width, with no gate half to
    split off or to put back.

    This has to be asked of the model, because the parameter cannot answer it:

    * the NAME cannot -- ``linear_fc1`` is the same name either way, and halving
      an ungated projection yields a tensor of exactly the right shape holding
      the wrong rows;
    * ``partition_stride`` cannot either, tempting as it looks. ``mlp.py``
      sets ``fc1_stride = 2`` for a gated dense MLP, but ``TEGroupedMLP``
      (``moe/experts.py``) doubles the width for gating and leaves stride at the
      TE default of 1, so a gated grouped expert reports stride 1.

    Both halves of the HF <-> Megatron round trip must ask this same question and
    get the same answer -- see Scripts-Slime/docs/06 section 9.
    """
    return bool(
        getattr(args, "gated_linear_unit", False)
        or getattr(args, "swiglu", False)
        or getattr(args, "quick_geglu", False)
    )
