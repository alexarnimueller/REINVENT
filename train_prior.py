#!/usr/bin/env python

import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem, rdBase
from tqdm import tqdm

from preprocess import MolData, Vocabulary
from model import RNN
from utils import decrease_learning_rate, mol_to_torchimage

rdBase.DisableLog('rdApp.error')


def pretrain(runname='chembl', restore_from=None):
    """Trains the prior RNN"""

    writer = SummaryWriter('logs/%s' % runname)

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc_%s" % runname)

    # Create a Dataset from a SMILES file
    moldata = MolData("data/mols_%s_filtered.smi" % runname, voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True, collate_fn=MolData.collate_fn)

    prior = RNN(voc)
    # writer.add_graph(prior.rnn, data.dataset[0])

    # Can restore from a saved RNN
    if restore_from:
        prior.rnn.load_state_dict(torch.load(restore_from))

    optimizer = torch.optim.Adam(prior.rnn.parameters(), lr=0.001)

    running_loss = 0.0
    for epoch in range(1, 6):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we don't overfit too much.
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader
            seqs = batch.long()

            # Calculate loss
            log_p, entropy = prior.likelihood(seqs)
            loss = -log_p.mean()
            running_loss += loss.item()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 250 steps we decrease learning rate and print some information
            if step % 250 == 249 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                seqs, likelihood, _ = prior.sample(128)
                valid = 0
                smiles = list()
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                        if valid < 5:
                            tqdm.write(smile)
                            smiles.append(smile.strip())

                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, running_loss / step))
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(prior.rnn.state_dict(), "data/prior_%s.ckpt" % runname)
                writer.add_scalar('training loss', running_loss / 250, epoch * len(data) + step)
                writer.add_scalar('valid_smiles', 100 * valid / len(seqs), epoch * len(data) + step)
                writer.add_image('sampled_mols', mol_to_torchimage(smiles))
                running_loss = 0.0

        # Save the prior and close writer
        torch.save(prior.rnn.state_dict(), "data/prior_%s.ckpt" % runname)
        writer.close()


if __name__ == "__main__":
    pretrain(sys.argv[1])
