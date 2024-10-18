#!/usr/bin/env python
import os, sys 

def train_flows(sample, hr=1): 
    cmd = 'python train_causalflow.py flow %s flow.%s -d /scratch/gpfs/chhahn/noah/floody/flow -v' % (sample, sample) 
    _deploy_base_slurm('flow.%s' % sample, 'o/flow.%s.o' % sample, cmd, hr=hr)
    return None


def train_supports(sample, hr=1): 
    cmd = 'python train_causalflow.py support %s support.%s -d /scratch/gpfs/chhahn/noah/floody/support -v' % (sample, sample) 
    _deploy_base_slurm('supp.%s' % sample, 'o/supp.%s.o' % sample, cmd, hr=hr)
    return None


def _deploy_base_slurm(job_name, output_name, cmd, hr=1): 
    '''
    '''
    script = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s" % job_name,
        "#SBATCH --nodes=1", 
        "#SBATCH --time=%s:59:59" % str(hr-1).zfill(2),
        "#SBATCH --mem=8G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=%s" % output_name, 
        "#SBATCH --mail-type=all",
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        cmd,
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the script.sh file, execute it and remove it
    f = open('script.slurm','w')
    f.write(script)
    f.close()
    os.system('sbatch script.slurm')
    os.system('rm script.slurm')
    return None 


if __name__=="__main__": 
    train_flows('treated', hr=6)
    train_flows('control', hr=6)
    train_supports('treated', hr=6)
    train_supports('control', hr=6)
