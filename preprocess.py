import re
import sys

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import Variable


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_length=140):
        self.special_token = ['EOS', 'GO']
        self.add_chars = set()
        self.chars = self.special_token
        self.vocab_size = len(self.chars)
        self.stoi = dict(zip(self.chars, range(len(self.chars))))
        self.itos = {v: k for k, v in self.stoi.items()}
        self.max_length = max_length
        if init_from_file:
            self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.stoi[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.stoi['EOS']:
                break
            chars.append(self.itos[i])
        return "".join(chars)

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        return split(smiles) + ['EOS']

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.add_chars.add(char)
        char_list = list(self.add_chars)
        char_list.sort()
        self.chars = char_list + self.special_token
        self.vocab_size = len(self.chars)
        self.stoi = dict(zip(self.chars, range(len(self.chars))))
        self.itos = {v: k for k, v in self.stoi.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters([c.strip() for c in chars])

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "SMILES vocabulary containing {} tokens: {}".format(len(self), self.chars)


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """

    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line.split()[0])

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        return Variable(encoded)

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""

    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)


def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=6, element_list=None):
    """Filters molecules on number of heavy atoms and atom types"""
    if not element_list:
        element_list = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    if mol is not None:
        num_heavy = min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        isotope = re.search(r'\[[1-9]', Chem.MolToSmiles(mol))
        if num_heavy and elements and not isotope:
            return True
        else:
            return False


def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")


def canonicalize_smiles_from_file(fname, molfilter=True):
    """Reads a SMILES file and returns a list of RDKit canonicalSMILES"""
    with open(fname, 'r') as f:
        out = []
        for i, line in enumerate(tqdm(f)):
            mol = Chem.MolFromSmiles(keep_longest(line.strip()))
            if mol:
                if molfilter:
                    if filter_mol(mol):
                        out.append(Chem.MolToSmiles(mol))
                else:
                    out.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(out)))
        return out


def keep_longest(smiles):
    """ function to keep the longest fragment of a smiles string after fragmentation by splitting at '.' """
    if '.' in smiles:
        f = smiles.split('.')
        lengths = [len(m) for m in f]
        return f[np.argmax(lengths)]
    else:
        return smiles


def split(smiles):
    pattern = r'\^|\s|#|=|-[0-9]*|\+[0-9]*|[0-9]|\[.{2,5}\]|%[0-9]{2}|\(|\)|\.|/|\\|:|@+|\{|\}|Cl|Ca|Cu|Br|Be|Ba|Bi|' \
              'Si|Se|Sr|Na|Ni|Rb|Ra|Xe|Li|Al|As|Ag|Au|Mg|Mn|Te|Zn|He|Kr|Fe|[BCFHIKNOPScnos]'
    return re.findall(pattern, smiles)


def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file."""
    add_chars = set()
    for i, smiles in enumerate(tqdm(smiles_list)):
        char_list = split(smiles)
        for char in char_list:
            add_chars.add(char)
    print("Number of characters: {}".format(len(add_chars)))
    with open('data/Voc', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars


if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_file(smiles_file, molfilter=True)
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(smiles_list)
    write_smiles_to_file(smiles_list, "data/mols_filtered.smi")
