#!/usr/bin/env python3
"""
Simple runner script for Category B methods.
Use this to test each method individually.
"""

import subprocess
import sys
from pathlib import Path

def run_method(method_name, cycle):
    """Run a specific method and cycle"""
    
    method_configs = {
        "ewc": f"ewc_head_expand/configs/catB_ewc_cycle{cycle}.yaml",
        "replay": f"experience_replay_head_expand/configs/catB_replay_cycle{cycle}.yaml", 
        "naive": f"naive_finetune/configs/catB_naive_cycle{cycle}.yaml",
        "isolation": f"parameter_isolation_new_branch/configs/catB_isolation_cycle{cycle}.yaml"
    }
    
    method_scripts = {
        "ewc": "ewc_head_expand/code/train_catB_ewc.py",
        "replay": "experience_replay_head_expand/code/train_catB_replay.py",
        "naive": "naive_finetune/code/train_catB_naive.py", 
        "isolation": "parameter_isolation_new_branch/code/train_catB_isolation.py"
    }
    
    if method_name not in method_configs:
        print(f"❌ Unknown method: {method_name}")
        print(f"Available: {list(method_configs.keys())}")
        return False
        
    config_path = Path(__file__).parent / method_configs[method_name]
    script_path = Path(__file__).parent / method_scripts[method_name]
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
        
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
        
    print(f"🚀 Running {method_name.upper()} Cycle {cycle}")
    print(f"   Config: {config_path}")
    print(f"   Script: {script_path}")
    print("-" * 50)
    
    try:
        cmd = [sys.executable, str(script_path), "--config", str(config_path)]
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {method_name.upper()} Cycle {cycle} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {method_name.upper()} Cycle {cycle} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {method_name.upper()} Cycle {cycle}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_catB.py <method> [cycle]")
        print("Methods: ewc, replay, naive, isolation")
        print("Cycles: 1, 2 (default: 1)")
        print("\nExamples:")
        print("  python run_catB.py ewc 1")
        print("  python run_catB.py replay 2") 
        print("  python run_catB.py naive")
        return
        
    method = sys.argv[1].lower()
    cycle = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    success = run_method(method, cycle)
    
    if success:
        print(f"\n🎉 {method.upper()} Cycle {cycle} training completed!")
        
        # Show output location
        method_dirs = {
            "ewc": "ewc_head_expand",
            "replay": "experience_replay_head_expand", 
            "naive": "naive_finetune",
            "isolation": "parameter_isolation_new_branch"
        }
        
        output_dir = Path(__file__).parent / method_dirs[method] / "outputs" / f"cycle{cycle}"
        print(f"📁 Results saved to: {output_dir}")
        
        if (output_dir / "metrics" / "metrics.json").exists():
            print("📊 Check metrics.json for detailed results")
        if (output_dir / "figures").exists():
            print("📈 Check figures/ for training curves and confusion matrices")
            
    else:
        print(f"\n💥 {method.upper()} Cycle {cycle} training failed!")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()