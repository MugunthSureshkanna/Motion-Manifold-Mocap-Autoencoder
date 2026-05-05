#!/bin/bash
#SBATCH -J mynotebook_job           # Job name
#SBATCH -o mynotebook_job.o%j       # Standard output file
#SBATCH -e mynotebook_job.e%j       # Standard error file
#SBATCH -p gpu-a100                 # Partition (choose as needed: gpu, normal, etc)
#SBATCH -N 1                        # Number of nodes
#SBATCH -n 1                        # Number of tasks
#SBATCH -t 00:20:00                 # Maximum runtime (hh:mm:ss)

module load python3/3.9.7

# Create virtual environment in $WORK directory (has more space than $HOME)
VENV_DIR="$WORK/my_venv"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio numpy scipy tqdm matplotlib opencv-python imageio imageio-ffmpeg jupyter ipython ipywidgets pillow

cd $SLURM_SUBMIT_DIR

# Run your script
python AE.py

# Deactivate virtual environment (optional)
deactivate