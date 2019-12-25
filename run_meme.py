import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy import log1p
import math
from random import random
from scipy.stats import norm
from tqdm import tqdm,tqdm_notebook
from scipy.stats.distributions import chi2
import difflib
from collections import defaultdict
from display import *
import warnings
warnings.filterwarnings("ignore")

elements=['A','C','G','T']
dict_map = { ele:i for i, ele in enumerate(elements)}
inverse_dict_map = { i:ele for i, ele in enumerate(elements)}


def main():
    parser = argparse.ArgumentParser(description='Estimate the start positions and motif model via MEME.')
    parser.add_argument('-ib', action="store", type=float, default='0.5',help="initial beta")
    parser.add_argument('-b', action="store", type=float, default='0.25', help="prior motif beta")
    parser.add_argument('-wmin', action="store", type=int, default='18',help="minimum width to search ")
    parser.add_argument('-wmax', action="store", type=int, default='18', help="maximum width to search")
    parser.add_argument('-m', action="store", type=str, default='ZOOPS', help="model type: OOPS or ZOOPS")
    parser.add_argument('-qt', action='store_false', default=True,
                        help='quick running TEST alhgorithm')
    parser.add_argument('-np', action='store', type=int,default=2,
                        help='number of passes of MEME')


    parser.add_argument('-f', action="store", type=str, default='lex.fa', help="sequence file name")
    args = parser.parse_args()


    param_dict = {'initial_beta': args.ib, 'beta': args.b, "Wmin": args.wmin, 'Wmax': args.wmax, 'model_type': args.m,
                  'quick_test': args.qt, 'npass': args.np}

    mp = MotifPreditor(args.f, **param_dict)

    mtruth = MotifTruth(args.f)

    mp.MEME()

    update_statistics(mp, mtruth)

    #print_meme_result(mp, mtruth)
    generate_latex(mp, mtruth, print_latex=False)

    return


if __name__ == '__main__':
    main()