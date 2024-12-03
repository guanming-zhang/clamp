#!/bin/bash
#SBATCH --job-name=ssl
#SBATCH --output=output_%j.log             
#SBATCH --error=error_%j.log               
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=2                
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:2                        
#SBATCH --time=1:00:00                        
#SBATCH --mem=6GB                 

## comments:
## see https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

## !must use srun to assign resources
## see https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html#troubleshooting
## troubleshooring section
srun singularity exec --nv --overlay /scratch/gz2241/sig-ml/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
            /bin/bash -c  "source /ext3/env.sh; 
            conda activate dl_env;
            python main.py /home/guanming/HPC-Scratch/sig-ml/clap/simulations/grid_search2/dir12#SSL#-lr-0.6-batch_size-128-lw0-1.0-lw2-1.0-#LC#-lr-0.2-run-00 ./default_config_cifar10.ini"
