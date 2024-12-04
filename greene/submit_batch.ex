#!/bin/bash
#SBATCH --job-name=ssl
#SBATCH --output=output_%j.log  # Save output to output_JOBID.log
#SBATCH --error=error_%j.log    # Save error to error_JOBID.log
#SBATCH --nodes=2               # Number of nodes (adjust as needed)
#SBATCH --ntasks-per-node=1     # One task per node
#SBATCH --cpus-per-task=16      # Number of CPUs per task (adjust as needed)
#SBATCH --gres=gpu:4            # Number of GPUs per node (adjust as needed)
#SBATCH --time=24:00:00         # Maximum runtime (adjust as needed)
#SBATCH --mem=64G               # Memory per node (adjust as needed)

# Load the required modules and activate Singularity
module load singularity  # Load Singularity module

# Path to your Singularity container
SINGULARITY_CONTAINER_PATH=/path/to/your_container.sif

# Execute your training script inside the Singularity container with Slurm's srun
srun singularity exec --nv $SINGULARITY_CONTAINER_PATH python train.py

singularity exec --nv --overlay /scratch/gz2241/sig-python/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c  "source /ext3/env.sh; 
            conda activate CONDA_ENV;
            python PYTHON_EXE ARGS"
