import torch
import numpy as np
from rdkit.Chem import Draw, MolFromSmiles, CanonSmiles


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles


def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if MolFromSmiles(smile):
            i += 1
    return i / len(smiles)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))


def is_valid_mol(smiles, return_smiles=False):
    """ function to check a generated SMILES string for validity """
    try:
        m = CanonSmiles(smiles.strip(), 1)
    except:
        m = None
    if return_smiles:
        return m is not None, m
    else:
        return m is not None


def mol_to_torchimage(smiles, labels=None):
    """ Function to plot 9 random molecules from a list of smiles for Tensorboard visualization """
    mols = [MolFromSmiles(s.strip()) for s in smiles]
    if any(labels):
        labels = [str(lab) for lab in labels]
    img = Draw.MolsToGridImage(mols=mols, legends=labels, molsPerRow=2, subImgSize=(600, 600))
    pic = np.array(img.getdata()).reshape((img.size[0], img.size[1], 3))
    img.close()
    pic = pic / 255.
    return torch.from_numpy(np.transpose(pic, (2, 0, 1)))
