#!/usr/bin/env python
import argparse
from agent import train_agent

parser = argparse.ArgumentParser(description="Main script for running the model")
parser.add_argument('--name', action='store', dest='runname', default='celecoxib', help='Name to identify the run.')
parser.add_argument('--prior', action='store', dest='priorname', default='chembl', help='Name of the prior to use.')
parser.add_argument('--scoring-function', action='store', dest='scoring_function',
                    choices=['NoSulphur', 'Tanimoto', 'ActivityModel'], default='Tanimoto',
                    help='What type of scoring function to use.')
parser.add_argument('--scoring-function-kwargs', action='store', dest='scoring_function_kwargs',
                    nargs="*", help='Additional arguments for the scoring function. Should be supplied with a '
                                    'list of "keyword_name argument". For pharmacophoric and tanimoto '
                                    'the keyword is "query" and requires a SMILES. '
                                    'For activity_model it is "clf_path " '
                                    'pointing to a sklearn classifier. '
                                    'For example: "--scoring-function-kwargs query COc1ccccc1".')
parser.add_argument('--learning-rate', action='store', dest='lr', type=float, default=0.0005)
parser.add_argument('--num-steps', action='store', dest='n_steps', type=int, default=3000)
parser.add_argument('--batch-size', action='store', dest='batch_size', type=int, default=64)
parser.add_argument('--sigma', action='store', dest='sigma', type=int, default=20)
parser.add_argument('--experience', action='store', dest='experience_replay', type=int,
                    default=0, help='Number of experience sequences to sample each step. '
                                    '0 means no experience replay.')
parser.add_argument('--num-processes', action='store', dest='num_processes', type=int, default=0,
                    help='Number of processes used to run the scoring function. "0" means '
                         'that the scoring function will be run in the main process.')
parser.add_argument('--save-dir', action='store', dest='save_dir',
                    help='Path where results and model are saved. Default is data/results/run_<datetime>.')

if __name__ == "__main__":
    arg_dict = vars(parser.parse_args())
    if arg_dict['scoring_function_kwargs']:
        kwarg_list = arg_dict.pop('scoring_function_kwargs')
        if not len(kwarg_list) % 2 == 0:
            raise ValueError("Scoring function kwargs must be given as pairs, got a list with odd length.")
        kwarg_dict = {i: j for i, j in zip(kwarg_list[::2], kwarg_list[1::2])}
        arg_dict['scoring_function_kwargs'] = kwarg_dict
    else:
        arg_dict['scoring_function_kwargs'] = dict()
    train_agent(**arg_dict)
