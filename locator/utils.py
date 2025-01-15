"""Utility functions for data processing"""

import numpy as np
from tqdm import tqdm

__all__ = [
    "load_genotypes",
    "sort_samples",
    "normalize_locs",
    "filter_snps",
]


def normalize_locs(locs):
    """Normalize location coordinates"""
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])
    locs = np.array(
        [[(x[0] - meanlong) / sdlong, (x[1] - meanlat) / sdlat] for x in locs]
    )
    return meanlong, sdlong, meanlat, sdlat, locs


def replace_md(genotypes):
    """Replace missing data with binomial draws from allele frequency"""
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(genotypes, min_mac=1, max_snps=None, impute=False):
    """Filter SNPs based on criteria"""
    print("filtering SNPs")
    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]

    if min_mac > 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = [x >= min_mac for x in derived_counts]
        genotypes = genotypes[ac_filter, :, :]

    if impute:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]

    if max_snps is not None:
        ac = ac[np.random.choice(range(ac.shape[0]), max_snps, replace=False), :]

    print("running on " + str(len(ac)) + " genotypes after filtering\n\n\n")
    return ac


def split_train_test(ac, locs, train_split=0.8):
    """Split data into training and test sets

    Args:
        ac: allele counts array
        locs: locations array
        train_split: proportion of data to use for training (default: 0.8)
    """
    train = np.argwhere(~np.isnan(locs[:, 0]))
    train = np.array([x[0] for x in train])
    pred = np.array([x for x in range(len(locs)) if x not in train])
    test = np.random.choice(train, round((1 - train_split) * len(train)), replace=False)
    train = np.array([x for x in train if x not in test])
    traingen = np.transpose(ac[:, train])
    trainlocs = locs[train]
    testgen = np.transpose(ac[:, test])
    testlocs = locs[test]
    predgen = np.transpose(ac[:, pred])
    return train, test, traingen, testgen, trainlocs, testlocs, pred, predgen
