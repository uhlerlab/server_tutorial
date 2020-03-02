import argparse


def setup_options():
    options = argparse.ArgumentParser()
    options.add_argument('-r', action='store', dest='seed',
                         default=3, type=int)
    return options.parse_args()
