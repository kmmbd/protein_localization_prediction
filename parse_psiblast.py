import os.path as osp
import numpy as np


def parse_psiblast(path_to_file):
    """
    For a concrete example check example.blastPsiMat format on
    https://github.com/plopd/ppcs2-project/blob/master/dataset/example.blastPsiMat
    :param path_to_file: 
    :return: 
    """

    # get filename (without extension) - it coincides with the identifier in the fasta format
    fn = osp.splitext(osp.split(path_to_file)[1])[0]
    f = open(path_to_file, "r")
    file = f.readlines()
    data = file[2:-6]
    f.close()
    record, sequence = [], []
    # get the first 20 amino acids
    aa = np.array(data[0].split()[:20])
    for line in data[1:]:
        values = line.split()
        results = list(map(int, values[22:42]))
        record.append(results)
        sequence.append(values[1])

    profile = np.array(record)
    return profile, sequence
