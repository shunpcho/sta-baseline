"""Learning rate policy."""

from fvcore.common.config import CfgNode


def get_lr_at_epoch(cfg: CfgNode, cur_epoch: float) -> float:
    """Retrieve the learning rate of the current epoch with the option to perform warm up in the beginning of the training stage.

    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.

    Returns:
        lr (float): the learning rate for the current epoch.
    """
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cur_epoch)
    # Perform warm up.
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
        lr_start = cfg.SOLVER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg, cfg.SOLVER.WARMUP_EPOCHS)
        alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start
    return lr


def lr_func_steps_with_relative_lrs(cfg: CfgNode, cur_epoch: float) -> float:
    """Retrieve the learning rate to specified values at specified epoch with the steps with relative learning rate schedule.

    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.

    Returns:
        lr (float): the learning rate for the current epoch.
    """
    ind = get_step_index(cfg, cur_epoch)
    return cfg.SOLVER.LRS[ind]


def get_step_index(cfg: CfgNode, cur_epoch: float) -> int:
    """Retrieves the lr step index for the given epoch.

    Args:
        cfg (CfgNode): configs. Details can be found in
            ego4d/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):  # NoQA
        if cur_epoch < step:
            break
    return max(ind - 1, 0)


def get_lr_func(lr_policy: str) -> callable:
    """Given the configs, retrieve the specified lr policy function.

    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError(f"Unknown LR policy: {lr_policy}")
    return globals()[policy]
