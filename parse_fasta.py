from Bio.SeqIO import parse
import pandas as pd
import numpy as np
import os.path as osp


def parse_fasta(path, fmt="fasta"):
    """
    
    :param fmt:
    :param path: 
    :return: 
    """

    d_fasta = {}
    for i, record in enumerate(parse(path, fmt)):
        id, label = record.description.split(" ")
        d_fasta[id] = label

    df = pd.DataFrame.from_dict(d_fasta, orient='index')
    df.columns = ['label']
    df['id'] = df.index
    df = df.set_index(np.arange(len(df)))

    return df

# if __name__ == '__main__':
#     fasta_path = osp.join('dataset', 'fasta_files', 'archaea.59.fa')
#     df1 = parse_fasta(fasta_path)
#     print(df1[:5])