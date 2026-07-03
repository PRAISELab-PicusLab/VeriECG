import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, CLIPECGTower


def build_ecg_tower(ecg_tower_cfg, **kwargs):
    model_name = getattr(ecg_tower_cfg, 'mm_ecg_tower', getattr(ecg_tower_cfg, 'ecg_tower', None))
    checkpoint_path = getattr(ecg_tower_cfg, 'mm_ecg_checkpoint', getattr(ecg_tower_cfg, 'ecg_checkpoint', None))

    print("=" * 60)
    print("INSIDE build_ecg_tower")
    print("CONFIG TYPE:", type(ecg_tower_cfg))
    print("MODEL:", repr(model_name))
    print("MODEL LEN:", len(model_name) if model_name else None)
    print("CKPT:", repr(checkpoint_path))
    print("EXISTS:", os.path.exists(checkpoint_path))
    print("=" * 60)

    if model_name == "ecg_coca" and os.path.exists(checkpoint_path):
        return CLIPECGTower(checkpoint_path, args=ecg_tower_cfg, **kwargs)

    raise ValueError(f"Unknown ecg tower: {model_name} | checkpoint: {checkpoint_path}")
    
def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
