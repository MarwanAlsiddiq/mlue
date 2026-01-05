#!/usr/bin/env python3
"""
Repository Cleanup Script for mlue project
Reorganizes files into proper directories and updates .gitignore
"""

import os
import shutil
from pathlib import Path

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'outputs/evaluation',
        'outputs/backtests',
        'outputs/models',
        'data/raw',
        'data/processed',
        'scripts',
        'notebooks',
        'docs'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"Created/verified: {d}/")

def move_files():
    """Move files to appropriate directories"""
    moves = {
        # Evaluation images
        'evaluation_bitcoin_confusion_matrix.png': 'outputs/evaluation/',
        'evaluation_bitcoin_roc_curve.png': 'outputs/evaluation/',
        'evaluation_gala_confusion_matrix.png': 'outputs/evaluation/',
        'evaluation_gala_roc_curve.png': 'outputs/evaluation/',
        
        # Backtest results
        'backtest_equity_curve.csv': 'outputs/backtests/',
        'walk_forward_preds.pkl': 'outputs/backtests/',
        
        # Scripts
        'kaggle_training_notebook.py': 'notebooks/',
        'cholestrol.py': 'scripts/',
        'choestrol.py': 'scripts/',  # Typo file
    }
    
    for src, dest_dir in moves.items():
        if os.path.exists(src):
            dest = os.path.join(dest_dir, os.path.basename(src))
            try:
                shutil.move(src, dest)
                print(f"Moved: {src} -> {dest}")
            except Exception as e:
                print(f"Error moving {src}: {e}")
        else:
            print(f"Not found: {src}")

def create_gitignore():
    """Create comprehensive .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints
*.ipynb

# Data files
*.csv
*.pkl
*.h5
*.hdf5
*.parquet
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Model files
*.pth
*.pt
*.ckpt
models/*.pth
models/*.pt
outputs/models/*
!outputs/models/.gitkeep

# Output files
*.png
*.jpg
*.jpeg
*.pdf
outputs/evaluation/*
!outputs/evaluation/.gitkeep
outputs/backtests/*
!outputs/backtests/.gitkeep

# Archives
*.zip
*.tar.gz
*.rar

# Logs
*.log
logs/

# Environment variables
.env
.env.local
secrets.yaml

# Kaggle
kaggle.json

# Temporary files
tmp/
temp/
*.tmp
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("Created/updated .gitignore")

def create_gitkeep_files():
    """Create .gitkeep files to preserve empty directories in git"""
    dirs = [
        'data/raw',
        'data/processed',
        'outputs/evaluation',
        'outputs/backtests',
        'outputs/models'
    ]
    
    for d in dirs:
        gitkeep = os.path.join(d, '.gitkeep')
        Path(gitkeep).touch()
        print("Created .gitkeep files")

def create_readme_structure():
    """Create README files for key directories"""
    readmes = {
        'outputs/README.md': """# Outputs Directory

This directory contains model outputs and results.

## Structure
- `evaluation/` - Model evaluation metrics and plots
- `backtests/` - Backtest results and equity curves
- `models/` - Saved model checkpoints
""",
        'data/README.md': """# Data Directory

Store your cryptocurrency data here.

## Structure
- `raw/` - Raw data from exchanges/APIs
- `processed/` - Preprocessed data ready for training
""",
        'scripts/README.md': """# Scripts Directory

Utility scripts for data processing, analysis, and maintenance.
"""
    }
    
    for path, content in readmes.items():
        with open(path, 'w') as f:
            f.write(content)
        print("Created directory READMEs")

def main():
    """Run all cleanup tasks"""
    print("Starting repository cleanup...\n")
    
    # Skip interactive prompt for automation
    print("Proceeding with cleanup...")
    
    print()
    create_directories()
    print()
    move_files()
    print()
    create_gitignore()
    create_gitkeep_files()
    create_readme_structure()
    
    print("\nCleanup complete!")
    print("\nNext steps:")
    print("1. Review the changes with: git status")
    print("2. Commit with: git add . && git commit -m 'Reorganize project structure'")

if __name__ == "__main__":
    main()