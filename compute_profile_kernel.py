import numpy as np
from itertools import product


def naive_profile_kernel_comp(seq, aa, profile, k=3, s=5, MAX_VALUE=1000):
    """
    Profile Kernel Calculation for a given sequence.
    Initialize variables and dataset structures.
    Count how often every k-mer $k_i$ appears in a protein $P$ with score $ \ge s$.
    :param seq: 
    :param aa: 
    :param profile: 
    :param k: 
    :param s: 
    :param MAX_VALUE: 
    :return: 
    """
    kmer_counts = np.zeros(len(aa)**k)  # initialize feature vector
    
    profile = profile / 100.  # transform profile scores to probabilities
    profile = np.where(profile == 0, MAX_VALUE, profile)  # avoid numerical instability
    # map values (0, 1] to -log() and 0 to MAX_VALUE
    profile = np.where(profile == MAX_VALUE, profile, -np.log(profile))

    kmer = list(product(aa, repeat=k))  # generate all aa^k possible k-mer
    kmer_index = list(product(np.arange(0, len(aa)), repeat=k))  # generate all possible tuples of length k

    # compute profile kernel naively
    for i, km in enumerate(kmer):
        # iterate with kmer using stepsize of 1 through sequence
        for j in range((len(seq) - k) + 1):
            # Get the profile score for `km`
            if seq[j:j+k] == list(km):
                rows = list(range(j, j+k))
                cols = list(kmer_index[i])
                res = np.sum(profile[rows, cols])
                # Count how often `km` appears in a protein whose score is larger than threshold `s`
                if res < s:
                    kmer_counts[i] += 1
    
    return kmer_counts
