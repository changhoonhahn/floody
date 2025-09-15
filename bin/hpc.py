'''

python script for deploying job on adroit 

'''
import os, sys
import time 


def floody(city, ncf=11, tag='v0', scenario='average', gpu=True, debug=True): 
    ''' run flood loss and crs saving calculations for specified city
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s.%s" % (city, tag),
        "#SBATCH -o o/%s.%s.o" % (city, tag), 
        ["#SBATCH --time=05:59:59", "#SBATCH --time=00:29:59"][debug], 
        "#SBATCH --export=ALL", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        ['', "#SBATCH --gres=gpu:1"][gpu], 
        ['', "#SBATCH --constraint=gpu80"][gpu], 
        "#SBATCH --mem-per-cpu=8G", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "floody=/home/chhahn/projects/floody/bin/run_floody.py",
        "python $floody %s -t %s -n %i -s %s" % (city, tag, ncf, scenario),
        "", 
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_floody.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _floody.slurm')
    os.system('rm _floody.slurm')
    return None 


if __name__=="__main__": 
    floody('houston', tag='v0_worst', ncf=11, scenario='max', gpu=False, debug=False) 
    floody('capecoral', tag='v0_worst', ncf=11, scenario='max', gpu=False, debug=False) 
    floody('chicago', tag='v0_worst', ncf=11, scenario='max', gpu=False, debug=False) 
    floody('losangeles', tag='v0_worst', ncf=11, scenario='max', gpu=False, debug=False) 
    floody('newyorkcity', tag='v0_worst', ncf=11, scenario='max', gpu=False, debug=False) 
    #floody('miami', tag='v0', ncf=11, gpu=False, debug=False) 
    #floody('neworleans', tag='v0', ncf=11, gpu=False, debug=False) 
