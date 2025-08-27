#!/bin/bash
#SBATCH -J finetune_job 
#SBATCH --gpus=4
#SBATCH --exclude=virya6
#SBATCH --exclude=virya5
#SBATCH -n20  # of CPU cores
#SBATCH --mem=64G
#SBATCH --time=150:00:00
#SBATCH -o %x-%j.out
#SBATCH --mail-type=BEGIN,END   # when to send email notication
#SBATCH --mail-user=rose.rostami@mail.concordia.ca 

# --nodelist=virya5

# Source the module binaries
source /etc/profile.d/modules.sh

# Load required modules
module load python/3.9.6
module load cuda/12.1.1
module load anaconda/3.2024.02

# Initialize Conda (this ensures Conda is correctly set up)
source /media/pkg/anaconda/v3.2024.02/root/etc/profile.d/conda.sh

# Activate your Conda environment
eval "$(conda shell.bash hook)"
conda activate sam2


# Print the current environment to verify
#echo "Current Conda environment: $CONDA_DEFAULT_ENV"

# create sub-directories, as needed
mkdir $TMPDIR/{src,dst}

#copy all input les/scripts to exec-host local disk
cp /home/g_rostam/omnicrack30k_rearranged.tar $TMPDIR/src
tar -xf $TMPDIR/src/omnicrack30k_rearranged.tar -C $TMPDIR/src
#cp /home/g_rostam/tiny_data.tar $TMPDIR/src
#tar -xf $TMPDIR/src/tiny_data.tar -C $TMPDIR/src



#echo $CUDA_VISIBLE_DEVICES

#echo $PATH
#echo $LD_LIBRARY_PATH
#which nvcc


python train_cluster_batch.py --slurm $TMPDIR/src

# copy output back to NFS
cp -r $TMPDIR/dst /home/g_rostam/sam2

