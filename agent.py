#!/usr/bin/env python

import datetime
import os
import time
from shutil import copyfile

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from model import RNN
from preprocess import Vocabulary, Experience
from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, mol_to_torchimage, is_valid_mol


def train_agent(runname='celecoxib', priorname='chembl', scoring_function='Tanimoto', scoring_function_kwargs=None,
                save_dir=None, batch_size=64, n_steps=3000, num_processes=6, sigma=60, experience_replay=5, lr=0.0005):
    print("\nStarting run %s with prior %s ..." % (runname, priorname))
    start_time = time.time()

    voc = Vocabulary(init_from_file="data/Voc_%s" % priorname)

    prior = RNN(voc)
    agent = RNN(voc)

    writer = SummaryWriter('logs/%s' % runname)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        prior.rnn.load_state_dict(torch.load('data/prior_%s.ckpt' % priorname))
        agent.rnn.load_state_dict(torch.load('data/prior_%s.ckpt' % priorname))
    else:
        prior.rnn.load_state_dict(torch.load('data/prior_%s.ckpt' % priorname, map_location=lambda storage, loc: storage))
        agent.rnn.load_state_dict(torch.load('data/prior_%s.ckpt' % priorname, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(agent.rnn.parameters(), lr=lr)

    # Scoring_function
    scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                            **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    print("Model initialized, starting training...")

    for step in range(n_steps):
        # Sample from Agent
        seqs, agent_likelihood, entropy = agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_ids = unique(seqs)
        seqs = seqs[unique_ids]
        agent_likelihood = agent_likelihood[unique_ids]
        entropy = entropy[unique_ids]

        # Get prior likelihood and score
        prior_likelihood, _ = prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles)

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience) > 4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        best_memory = experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h\n".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i], prior_likelihood[i],
                                                                       augmented_likelihood[i], score[i], smiles[i]))

        # Log
        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('score', np.mean(score), step)
        writer.add_scalar('entropy', entropy.mean(), step)
        if best_memory:
           writer.add_scalar('best_memory', best_memory, step)

        # get 4 random valid smiles and scores for logging
        val_ids = np.array([i for i, s in enumerate(smiles) if is_valid_mol(s)])
        val_ids = np.random.choice(val_ids, 4, replace=False)
        smiles = np.array(smiles)[val_ids]
        score = ['%.3f' % s for s in np.array(score)[val_ids]]
        writer.add_image('generated_mols', mol_to_torchimage(smiles, score), step)

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        save_dir = 'results/%s' % runname + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir)
    copyfile('agent.py', os.path.join(save_dir, "agent_%s.py" % runname))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(agent.rnn.state_dict(), os.path.join(save_dir, 'Agent_%s.ckpt' % runname))

    seqs, agent_likelihood, entropy = agent.sample(256)
    prior_likelihood, _ = prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score = scoring_function(smiles)
    with open(os.path.join(save_dir, "sampled.txt"), 'w') as f:
        f.write("SMILES Score PriorLogP\n")
        for smiles, score, prior_likelihood in zip(smiles, score, prior_likelihood):
            f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score, prior_likelihood))

    print("\nDONE! Whole run took %s" % datetime.timedelta(seconds=time.time()-start_time))


if __name__ == "__main__":
    train_agent()
