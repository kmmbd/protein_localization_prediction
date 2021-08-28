import os
import numpy as np
import pandas as pd
import os.path as osp
import time
import argparse

from parse_fasta import parse_fasta
from parse_psiblast import parse_psiblast
from compute_profile_kernel import naive_profile_kernel_comp
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing
from env_paths import path_exists

# utils
ROOT_DIR = os.getcwd()
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Dataset Preparation for Sub-cellular Localization with a Classifier')
parser.add_argument('--fasta_in', default='eukaryota.1682.fa', type=str, help='FASTA input file')
parser.add_argument('--save_pth', default='dataset/subloc/', type=str, help='Path where to save final data')
parser.add_argument('--profile_pth', default='dataset/profiles/', type=str, help='PSI-Blast output file path(s)')
parser.add_argument('--k', default=3, type=int, help='Kmer size')
parser.add_argument('--sigma', default=5, type=int, help='Kmer frequency threshold for the Profile Kernel')
parser.add_argument('--ths_cls', default=10, type=int, help='Threshold to remove datapoints belonging to low-frequency classes')
parser.add_argument('--num_samples', default=None, type=int, help='Number of samples stored in final data')
parser.add_argument('--ths_kmer', default=0, type=int, help='Threshold to remove datapoints with low sum of kmer frequency ')
args = parser.parse_args()


def make_data(fasta_in, save_pth, profile_pth, k, sigma, ths_cls, num_samples, ths_kmer):

    start_time = time.time()
    print('Start making data...')
    # Parse FASTA format files
    pth = osp.join('dataset', 'fasta_files')
    path_exists(pth)
    df1 = parse_fasta(osp.join(pth, fasta_in))

    # compute kmer frequency feature vector for each datapoint in `df1`
    id2kmer_counts = {}
    N = df1.shape[0]
    pbar = tqdm(range(N))
    path_exists(profile_pth)
    for i in pbar:
        if num_samples is not None:
            # consider only first `num_samples`
            if i == num_samples:
                break
        _, idx = df1.iloc[i, 0], df1.iloc[i, 1]
        pbar.set_description("Processing input {}: {:d}/{:d} ".format(idx, i, N))
        fp = osp.join(ROOT_DIR, profile_pth, idx + '.blastPsiMat')
        if os.path.isfile(fp):
            profile, seq = parse_psiblast(fp)
            id2kmer_counts[idx] = naive_profile_kernel_comp(seq, amino_acids, profile, k=k, s=sigma)

    # create dataframe, shape=[N, M], N = datapoints, M = kmers
    df2 = pd.DataFrame.from_dict(id2kmer_counts, orient='index')
    df2.columns = [k * "%s" % x for x in list(product(amino_acids, repeat=k))]
    df2['id'] = df2.index
    df2 = df2.set_index(np.arange(len(df2)))

    # merge `df1` with `df2` to `df3`, shape=[N, M + 2]
    df3 = pd.merge(df1, df2, on='id')
    print('Data shape: {}'.format(df3.shape))

    # # transform labels to numerical values
    # y = df3['label'].values
    # le = preprocessing.LabelEncoder()
    # le.fit(y)
    # y = le.transform(y)
    #
    # ######### START PRE-PROCESSING ##########
    # print('Start preprocessing data...')
    # # STEP 1
    # # remove datapoints belonging to classes with frequency less than `ths_cls`
    # _, counts = np.unique(y, return_counts=True)
    # mask = np.isin(y, np.where(counts >= ths_cls)[0])
    # df3 = df3[mask]
    # print('After removing datapoints belonging to low-frequency classes, new data shape: {}'.format(df3.shape))
    #
    # # STEP 2
    # # remove datapoints where kmer frequency sum is less or equal to `ths_kmer`
    # mask = np.sum(df3.iloc[:, 2:].values, axis=1) > ths_kmer
    # df3 = df3[mask]
    # print('After removing datapoints having sum of kmer frequency below threshold, new data shape: {}'.format(df3.shape))
    # print('Finished pre-processing data...')
    # ######### END PRE-PROCESSING ##########

    end_time = time.time()
    print('Finished making data in: {:.3f} seconds'.format(end_time - start_time))

    # save csv data to disk
    identifier = str(end_time).replace(".", "")
    path_exists(save_pth)
    df3.to_csv(osp.join(ROOT_DIR, save_pth, 'subloc_k{}_s{}_{}_{}.csv'.format(k, sigma, fasta_in, identifier)), index=False)
    # df3.loc[:, df3.columns == 'id'].to_csv(osp.join(ROOT_DIR, save_pth, 'subloc_k{}_s{}_ths_cls_{}_ths_kmer_{}_{}_{}_id.csv'.format(k, sigma, ths_cls, ths_kmer, fasta_in, identifier)), index=False)
    # df3.loc[:, df3.columns != 'id'].to_csv(osp.join(ROOT_DIR, save_pth, 'subloc_k{}_s{}_ths_cls_{}_ths_kmer_{}_{}_{}_no_id.csv'.format(k, sigma, ths_cls, ths_kmer, fasta_in, identifier)), index=False)


if __name__ == '__main__':
    make_data(fasta_in=args.fasta_in,
              save_pth=args.save_pth,
              profile_pth=args.profile_pth,
              k=args.k, sigma=args.sigma,
              ths_cls=args.ths_cls,
              num_samples=args.num_samples,
              ths_kmer=args.ths_kmer)
