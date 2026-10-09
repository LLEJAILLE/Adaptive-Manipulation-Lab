"""One lazy command dispatcher; importing the CLI never opens a viewer."""

import argparse
from importlib import import_module

COMMANDS = {
    'manual': ('adaptive_manipulation.interfaces.manual', 'Piloter le robot avec la manette et les curseurs'),
    'record': ('adaptive_manipulation.workflows.record', 'Enregistrer des demonstrations a la manette'),
    'inspect-demos': ('adaptive_manipulation.data.demonstrations', 'Verifier un dataset de demonstrations'),
    'train-bc': ('adaptive_manipulation.workflows.train_bc', 'Entrainer l Actor par imitation'),
    'train-sac': ('adaptive_manipulation.workflows.train_sac', 'Entrainer SAC, initialiser depuis BC ou reprendre un run'),
    'evaluate': ('adaptive_manipulation.workflows.evaluate', 'Evaluer un checkpoint BC ou SAC dans MuJoCo'),
    'plot': ('adaptive_manipulation.analysis.plot_runs', 'Tracer les CSV d une experience BC ou SAC'),
}

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m adaptive_manipulation',
                                     description='Adaptive Manipulation Lab : simulation, demonstrations, BC et SAC')
    commands = parser.add_subparsers(dest='command',required=True)
    for command,(_,description) in COMMANDS.items():
        commands.add_parser(command,help=description,add_help=False)
    options,remaining = parser.parse_known_args(argv)
    return import_module(COMMANDS[options.command][0]).main(remaining)
