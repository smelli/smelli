import argparse
from smelli.unbinned import DataSets
from smelli.util import get_datapath
import os

def cli():
    parser = argparse.ArgumentParser(description='Command line interface to interact with smelli')
    subparsers = parser.add_subparsers(title='subcommands')

    # download
    parser_download = subparsers.add_parser('download',
                                            description='Command line script to download an additional data set',
                                            help='download an additional data set')
    parser_download.set_defaults(func=do_download)
    parser_download.add_argument('dataset',
                                 metavar='DATASET',
                                 type=str,
                                 help='the dataset identifier')

    # list
    parser_list = subparsers.add_parser('list',
                                        description='Command line script to list all known additional data sets',
                                        help='list all known additional data sets')
    parser_list.set_defaults(func=do_list)

    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        parser.print_help()

def do_download(args):
    path = get_datapath('smelli', 'data/yaml/datasets.yaml')
    with open(path) as f:
        D = DataSets.load(f)
    D.download(args.dataset)

def do_list(args):
    path = get_datapath('smelli', 'data/yaml/datasets.yaml')
    with open(path) as f:
        D = DataSets.load(f)
    for dn in D:
        print(dn)
