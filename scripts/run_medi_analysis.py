import os
import sys
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from medi_pipeline.config import paths

FEATURES_DIR = getattr(paths, 'FEATURES_DIR')
MEDI_SCRIPT = os.path.join(FEATURES_DIR, 'medi_analysis.py')

def run_script(script_path, description):
    print(f"--- Running {description} ---")
    print(f"Script: {script_path}")
    
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if str(project_root) not in pythonpath:
         env['PYTHONPATH'] = str(project_root) + os.pathsep + pythonpath

    try:
        subprocess.run([sys.executable, str(script_path)], check=True, env=env)
        print(f"=== {description} Completed Successfully ===\n")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {description}: {e} !!!")
        sys.exit(1)

def main():
    run_script(MEDI_SCRIPT, "MEDI Analysis")
    print("MEDI analysis completed.")

if __name__ == "__main__":
    main()
