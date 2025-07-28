#!/bin/bash
#SBATCH --job-name=ssl-detection
#SBATCH --output=output_%j.log             
#SBATCH --error=error_%j.log               
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1         
#SBATCH --cpus-per-task=12      
#SBATCH --gres=gpu:4                        
#SBATCH --time=47:30:00                      
#SBATCH --mem=120GB 
                

## comments:
## see https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html

## !must use srun to assign resources
## see https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html#troubleshooting
## troubleshooring section
## To add constraints for the machines
##SBATCH --constraint=rtx8000|a100
##SBATCH --partition=chemistry_a100_2
##SBATCH --nodelist=ga029,ga031
##SBATCH --constraint=a100

srun singularity exec --nv --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro\
            --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro\
            --overlay /scratch/gz2241/sig-ml/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif\
            /bin/bash -c  "source /ext3/env.sh; 
            conda activate dl_env;
            python train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco_lr_0.18.yaml \
	--dist-url tcp://127.0.0.1:48000 \--num-gpus 4   --num-machines 1 --machine-rank 0 \
	MODEL.WEIGHTS  resnet50_4views.pkl OUTPUT_DIR ./output_lr_0.18" 
