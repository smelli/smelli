#! /usr/bin/env python3

"""Standalone helper script to re-generate the SM covariance data files."""

import argparse
import logging
import sys
logging.basicConfig(level=logging.INFO)


def main(argv):
    parser = argparse.ArgumentParser(description='Recompute and save the SM covariances to the working directory.')
    parser.add_argument('-n', type=int, default=100,
                        help='Number of evaluations (default 100)')
    parser.add_argument('-t', type=int, default=1,
                        help='Number of threads (default 1)')
    parser.add_argument('-f', action='store_true',
                        help='Force recomputation (default false)')

    args = parser.parse_args()

    from smelli import GlobalLikelihood
    gl = GlobalLikelihood()

    logging.info("Computing covariances with N={} and {} threads".format(args.n, args.t))

    gl.make_measurement(N=args.n, threads=args.t, force=args.f)
    gl.save_sm_covariances('.')


if __name__ == '__main__':
    main(sys.argv)
