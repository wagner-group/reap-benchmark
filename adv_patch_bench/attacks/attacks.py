"""Setup all attack algorithms."""
from typing import Any, Dict, Optional, Tuple

import torch
from adv_patch_bench.attacks import base_attack
from adv_patch_bench.attacks.rp2 import rp2_detectron, rp2_yolo

_ATTACK_DICT = {
    "rp2-detectron": rp2_detectron.RP2AttackDetectron,
    "rp2-yolo": rp2_yolo.RP2AttackYOLO,
}


def setup_attack(
    config_attack: Optional[Dict[Any, str]] = None,
    is_detectron: bool = True,
    model: Optional[torch.nn.Module] = None,
    input_size: Tuple[int, int] = (1536, 2048),
    verbose: bool = False,
) -> base_attack.DetectorAttackModule:
    """Set up attack object."""
    # TODO(feature): Add no_attack as an attack option.
    attack_name: str = config_attack["common"]["attack_name"]
    if is_detectron:
        attack_fn_name: str = f"{attack_name}-detectron"
    else:
        attack_fn_name: str = f"{attack_name}-yolo"
    attack_fn = _ATTACK_DICT[attack_fn_name]
    combined_config_attack: Dict[str, Any] = {**config_attack["common"], 
                                              **config_attack[attack_name]}

    return attack_fn(
        combined_config_attack,
        model,
        input_size=input_size,
        verbose=verbose,
    )
