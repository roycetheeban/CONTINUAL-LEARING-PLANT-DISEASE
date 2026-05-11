#!/usr/bin/env python3
"""
Test script to verify Category B setup is working correctly.
Run this before starting actual training to catch any issues early.
"""

import sys
from pathlib import Path
import yaml
import torch

# Add shared code to path
sys.path.append(str(Path(__file__).resolve().parent / '_shared' / 'code'))

def test_config_loading():
    """Test that all config files can be loaded"""
    print("Testing config loading...")
    
    configs = [
        "ewc_head_expand/configs/catB_ewc_cycle1.yaml",
        "ewc_head_expand/configs/catB_ewc_cycle2.yaml",
        "experience_replay_head_expand/configs/catB_replay_cycle1.yaml", 
        "experience_replay_head_expand/configs/catB_replay_cycle2.yaml",
        "naive_finetune/configs/catB_naive_cycle1.yaml",
        "naive_finetune/configs/catB_naive_cycle2.yaml",
        "parameter_isolation_new_branch/configs/catB_isolation_cycle1.yaml",
        "parameter_isolation_new_branch/configs/catB_isolation_cycle2.yaml"
    ]
    
    for config_path in configs:
        full_path = Path(__file__).parent / config_path
        if not full_path.exists():
            print(f"❌ Missing config: {config_path}")
            continue
            
        try:
            cfg = yaml.safe_load(full_path.read_text(encoding='utf-8'))
            print(f"✅ {config_path}")
            
            # Check required fields
            required = ['seed', 'device', 'model', 'data', 'classes', 'train', 'output']
            missing = [f for f in required if f not in cfg]
            if missing:
                print(f"   ⚠️  Missing fields: {missing}")
                
        except Exception as e:
            print(f"❌ Error loading {config_path}: {e}")

def test_common_imports():
    """Test that common_catb.py imports work"""
    print("\nTesting common imports...")
    
    try:
        from common_catb import (
            set_seed, ensure_dirs, build_transforms, build_global_classes, 
            RemapImageFolder, load_base_model, expand_head_to_7, freeze_for_catb,
            train_epoch_mixed, compute_fisher, build_theta_star
        )
        print("✅ All common imports successful")
        
        # Test basic functionality
        set_seed(42)
        print("✅ set_seed works")
        
        # Test model loading (dummy)
        try:
            from torchvision import models
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[3] = torch.nn.Linear(1024, 5)
            expand_head_to_7(model, 7)
            assert model.classifier[3].out_features == 7
            print("✅ expand_head_to_7 works")
            
            freeze_for_catb(model, unfreeze_g2=True)
            print("✅ freeze_for_catb works")
            
        except Exception as e:
            print(f"⚠️  Model operations test failed: {e}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")

def test_data_paths():
    """Test that data paths in configs exist"""
    print("\nTesting data paths...")
    
    # Test one config as example
    config_path = Path(__file__).parent / "ewc_head_expand/configs/catB_ewc_cycle1.yaml"
    if not config_path.exists():
        print("❌ Cannot test data paths - config missing")
        return
        
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    data_paths = [
        cfg['data']['old_train_dir'],
        cfg['data']['old_replay_dir'], 
        cfg['data']['new_train_dir'],
        cfg['data']['old_val_dir'],
        cfg['data']['new_val_dir'],
        cfg['data']['old_test_dir'],
        cfg['data']['new_test_dir']
    ]
    
    for path_str in data_paths:
        path = Path(path_str)
        if path.exists():
            print(f"✅ {path_str}")
        else:
            print(f"❌ Missing: {path_str}")
            
    # Check critical research logic alignment
    print("\n🔬 Research Logic Checks:")
    
    # Check EWC has old_f1_drop_tolerance
    if 'old_f1_drop_tolerance' in cfg.get('train', {}):
        print("✅ EWC early stopping logic present")
    else:
        print("❌ Missing old_f1_drop_tolerance for early stopping")
        
    # Check split-LR parameters
    train_cfg = cfg.get('train', {})
    if 'lr_head_old' in train_cfg and 'lr_head_new' in train_cfg:
        print("✅ Split-LR classifier parameters present")
        print(f"   Old LR: {train_cfg['lr_head_old']}, New LR: {train_cfg['lr_head_new']}")
    else:
        print("❌ Missing split-LR classifier parameters")

def test_base_checkpoint():
    """Test that base checkpoint exists"""
    print("\nTesting base checkpoint...")
    
    config_path = Path(__file__).parent / "ewc_head_expand/configs/catB_ewc_cycle1.yaml"
    if not config_path.exists():
        print("❌ Cannot test checkpoint - config missing")
        return
        
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    ckpt_path = Path(cfg['model']['base_checkpoint'])
    
    if ckpt_path.exists():
        print(f"✅ Base checkpoint exists: {ckpt_path}")
        
        # Try loading it
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            print(f"✅ Checkpoint loads successfully")
            print(f"   Keys: {len(state)} parameters")
        except Exception as e:
            print(f"❌ Checkpoint load error: {e}")
    else:
        print(f"❌ Missing base checkpoint: {ckpt_path}")

def main():
    print("🧪 Category B Setup Test")
    print("=" * 50)
    
    test_config_loading()
    test_common_imports() 
    test_data_paths()
    test_base_checkpoint()
    
    print("\n" + "=" * 50)
    print("✅ Setup test complete!")
    print("\nIf you see any ❌ errors above, fix them before training.")
    print("If all ✅, you're ready to start Category B training!")

if __name__ == "__main__":
    main()