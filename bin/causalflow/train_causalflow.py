#!/usr/bin/env python
import argparse
import os 
import numpy as np

import torch
import optuna 
# floody
from floody import data as D
# causalflow
from causalflow import causalflow



def FlowObjective(trial):
    ''' bojective function for optuna
    '''
    # Generate the model
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True)
    n_comp = trial.suggest_int("n_comp", n_comp_min, n_comp_max)
    
    if args.verbose: 
        print('n_blocks = %i, n_hidden = %i' % (n_blocks, n_hidden))
        print('n_transf = %i, n_comp = %i, learning_rate = %f' % (n_transf, n_comp, lr))

    Cflow.set_architecture(
            arch='made',
            nhidden=n_hidden,
            ntransform=n_transf,
            nblocks=n_blocks,
            num_mixture_components=n_comp,
            batch_norm=True)


    flow, best_valid_log_prob = Cflow._train_flow(XY[:,0], XY[:,1:],
           outcome_range=[[-1.], [6.]],
           training_batch_size=50,
           learning_rate=lr,
           verbose=False)

    # save trained NPE
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    if args.verbose: 
        print('saving %s' % fflow) 
    torch.save(flow, fflow)

    return -1*best_valid_log_prob


def SupportObjective(trial):
    ''' bojective function for optuna
    '''
    # Generate the model
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True)
    if args.verbose: print('n_blocks = %i, n_hidden = %i, learning_rate = %f' % 
                           (n_blocks, n_hidden, lr))

    # set architecture
    Sup.set_architecture(ndim,
            nhidden=n_hidden,
            nblock=n_blocks)

    # run trianing
    flow, best_valid_loss = Sup._train(XY[:,1:],
            batch_size=50,
            learning_rate=lr,
            num_iter=300,
            clip_max_norm=1,
            verbose=False)

    # save trained flow
    fflow = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(flow, fflow)

    return best_valid_loss


if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("flow", help="flow or support") 
    parser.add_argument("sample", help='treated or control sample') 
    parser.add_argument("study", help="optuna study")
    parser.add_argument("-d", "--dir", help="output directory")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = ("cuda:0" if cuda else "cpu")

    study_name = args.study # optuna study name 
    output_dir = args.dir 

    if args.flow == 'flow': # train flow
        Cflow = causalflow.CausalFlowA(device=device)
    elif args.flow == 'support': # train support flow 
        Sup = Support.Support(device=device)
    else: 
        raise ValueError

    # read data 
    fema = D.FEMA()
    XY, _ = fema.prepare_train_test(args.sample, split=0.9, seed=42)
    # reduce dynamic scale
    XY[:,0] = np.log10(XY[:,0])
    XY[:,3] = np.log10(XY[:,3])
    XY[:,4] = np.log10(XY[:,4])
    if args.verbose: print('Ntrain = %i' % (XY.shape[0]))

    # Optuna Parameters
    n_trials   = 1000
    n_jobs     = 1
    if not os.path.isdir(os.path.join(output_dir, study_name)):
        os.system('mkdir %s' % os.path.join(output_dir, study_name))
    storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
    n_startup_trials = 20

    n_blocks_min, n_blocks_max = 2, 5
    n_transf_min, n_transf_max = 2, 5
    n_hidden_min, n_hidden_max = 32, 128
    n_comp_min, n_comp_max = 1, 5
    n_lr_min, n_lr_max = 5e-6, 1e-3

    sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
    study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True)

    if args.flow == 'flow': # train flow
        study.optimize(FlowObjective, n_trials=n_trials, n_jobs=n_jobs)
    elif args.flow == 'support': # train support flow 
        study.optimize(SupportObjective, n_trials=n_trials, n_jobs=n_jobs)
    print("  Number of finished trials: %i" % len(study.trials))
