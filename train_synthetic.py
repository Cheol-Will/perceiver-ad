import torch
import numpy as np
import argparse
import os
import json
from DataSet.data_generator import DataGenerator


def main():
    dg = DataGenerator()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--r')
    
    args = parser.parse_args()
    main(args)