import sys, msprime as msp, numpy as np
import multiprocessing as mp, subprocess as sp
import pyslim, os, io, shutil
import shlex, scipy, time, re
from shutil import copyfile
import allel
import time
from scipy import spatial
from scipy import stats
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

##debug params
# ts=pyslim.load("/Users/cj/spaceness/sims/slimout/spatial/W50/coalesced/sigma_0.5187220106279465_.trees1500000.trees")
# ts=sample_treeseq(infile=ts,nSamples=100,outfile="",recapitate=False,recombination_rate=1e-8,write_to_file=False)
# ts=msp.mutate(ts,1e-8)
# haps,positions,locs=get_ms_outs(ts)
# label=float(re.sub("sigma_|_.trees*|.trees","",os.path.basename("/Users/cj/spaceness/sims/slimout/spatial/W50/coalesced/sigma_0.5187220106279465_.trees1500000.trees")))
# maxlen=1e8

def run_one_slim_sim(sampled_param,
                     min,
                     max,
                     slim_path,
                     slim_recipe,
                     outdir):
    '''
    Run one SLiM simulation pulling parameter values from a uniform
    distribution. Output will be named with the name and value of
    the sampled parameter.

    See run_slim.py for a version of this function that accepts command-line
    arguments.

    '''

    #get sampled param names and values
    val=np.random.uniform(min,max)

    #get output file path
    filename = sampled_param+"_"+str(val)+"_.trees"
    filepath = os.path.join(outdir,filename)

    #set strings for defining SLiM variables
    label_str=sampled_param+"="+str(val)
    output_str = "outpath='"+str(filepath)+"'"

    #format params for subprocess.check_output
    command=[slim_path,
             "-d",output_str,
             "-d",label_str,
             slim_recipe]

    #run it
    print("starting slim simulation with "+args.sampled_param+"="+str(val))
    sp.check_output(command)

    return None

def check_treeseq_coalescence(indir):
    '''
    Read in a set of SLiM tree sequences and record (1) the
    proportion of uncoalesced gene trees and (2) the mean
    number of remaining lineages at the root.
    '''
    treeseqs=[f for f in os.listdir(indir) if not f.startswith(".")]
    out=np.zeros(shape= (2,len(treeseqs)))
    j=0
    for t in tqdm(treeseqs):
        ts=pyslim.load(os.path.join(indir,t))
        sts=ts.simplify()
        nroots=[0 for _ in range(sts.num_trees)]
        i=0
        for t in sts.trees():
            nroots[i]=t.num_roots
            i=i+1
        prop_uncoalesced=np.sum([x>1 for x in nroots])/len(nroots)
        mean_lineages=np.mean(nroots)
        out[:,j]=[prop_uncoalesced,mean_lineages]
        j=j+1
    return treeseqs,out

def get_pop_sizes(indir,outpath):
    trees = trees=[f for f in os.listdir(indir) if not f.startswith(".")]
    labels=[float(re.sub("sigma_|_.trees*|.trees","",x)) for x in trees]
    out=np.empty(shape=(len(trees),2))
    for i in tqdm(range(len(trees))):
        ts=pyslim.load(os.path.join(indir,trees[i]))
        popsize=ts.num_samples/2
        out[i]=[labels[i],popsize]
    np.savetxt(outpath,out)
    return out

def sample_treeseq_directory(indir,outdir,nSamples,recapitate,recombination_rate):
    '''
    Sample individuals from all tree sequences a directory then simplify and (optionally)
    recapitate the treeseq_to_vcf.py with msprime.
    '''
    trees=[f for f in os.listdir(indir) if not f.startswith(".")]
    #pop_sizes=[]
    for i in tqdm(range(len(trees))):
        ts=pyslim.load(os.path.join(indir,trees[i]))
        #if(output_pop_sizes):
            #pop_sizes.append(ts.num_samples/2)
        sample_inds=np.unique([ts.node(j).individual for j in ts.samples()]) #final-generation individuals
        subsample=np.random.choice(sample_inds,nSamples,replace=False) #get nSamples random sample inds
        subsample_nodes=[ts.individual(x).nodes for x in subsample] #node numbers for sampled individuals
        subsample_nodes=[a for b in subsample_nodes for a in b] #flatten the list
        subsample_nodes=np.sort(np.array(subsample_nodes))
        o=ts.simplify(subsample_nodes)
        if(recapitate):
            ts=ts.recapitate(recombination_rate=recombination_rate)
        o.dump(os.path.join(outdir,trees[i]))
    #if(output_pop_sizes):
    #    np.savetxt(pop_size_outpath,pop_sizes)
    return None

def sample_treeseq(infile,
                   nSamples,
                   sampling,
                   recapitate=False,
                   recombination_rate=1e-9,
                   write_to_file=False,
                   outfile='',
                   sampling_locs=[],
                   plot=False,
                   seed=0,
                   dist_scaling=3):
    '''
    Samples a set of individuals from a SLiM tree sequence and returns a
    tree sequence simplified for those individuals.

    params:
    infile - string or object. input treesequence file or object
    nSamples - integer. total samples to return
    sampling - 'random' (nSamples random individuals);
               'midpoint' (nSamples closest individuals to the middle of the landscape);
               'point' (nSamples/len(sampling_locs) individuals closest to each location in sampling_locs).
    recapitate - boolean. Should simulations be run to coalescence with msprime? False unless you know *exactly* what you're doing here.
    recombination_rate - float. Recombination rate to use for recapitation, in recombination events per base per generation.
    write_to_file - boolean.
    outfile - string. File path for output.
    sampling_locs - list. locations to draw samples from when using sampling='point', as [[x1,y1],[x2,y2],...]
    plot - boolean. plot location of samples?
    seed - random seed for all numpy operations. 0 uses the system default.
    '''
    if type(infile)==str:
        ts=pyslim.load(infile)
    else:
        ts=infile
    if(not seed==0):
        np.random.seed(seed)
    if(recapitate):
        ts=ts.recapitate(recombination_rate=recombination_rate) #moved from bottom bc simplify converts to msprime treeseq(??? - ask Peter)
    if(sampling=="random"):
        sample_inds=np.unique([ts.node(j).individual for j in ts.samples()]) #final-generation individuals
        subsample=np.random.choice(sample_inds,nSamples,replace=False) #get nSamples random sample inds
        subsample_nodes=[ts.individual(x).nodes for x in subsample] #node numbers for sampled individuals
        subsample_nodes=[a for b in subsample_nodes for a in b] #flatten the list
        subsample_nodes=np.sort(np.array(subsample_nodes))
        o=ts.simplify(subsample_nodes)
        locs=np.array([[x.location[0],x.location[1]] for x in ts.individuals()])
        olocs=np.array([[x.location[0],x.location[1]] for x in o.individuals()])
        if plot:
            plt.scatter(locs[:,0], locs[:,1],s=80,facecolors='none',edgecolors='k')
            plt.scatter(olocs[:,0],olocs[:,1],s=80,facecolors='r',edgecolors='r')
            plt.axis([0, np.max(locs[:,0]), 0, np.max(locs[:,1])])
    if(sampling=="midpoint"):
        #get final generation individuals
        extant_inds=np.unique([ts.node(j).individual for j in ts.samples()]) #final-generation individuals
        extant_nodes=[ts.individual(x).nodes for x in extant_inds] #node numbers for final-generation individuals
        extant_nodes=[a for b in extant_nodes for a in b]
        ts=ts.simplify(extant_nodes)
        #sample nSamples individuals proportional to their distance from the midpoint where p(sample)=(1/dist**4)/sum(dists_to_middle)
        locs=np.array([[x.location[0],x.location[1]] for x in ts.individuals()])
        inds=np.array([x.id for x in ts.individuals()])
        middle=[np.mean(locs[:,0]),np.mean(locs[:,1])]
        dists_to_middle=[scipy.spatial.distance.euclidean(locs[x],middle) for x in range(len(locs))]
        weights=[1/x**4 for x in dists_to_middle]
        weights=weights/np.sum(weights)
        subsample=np.random.choice(inds,nSamples,p=weights,replace=False)
        #closest_inds=sorted(range(len(dists_to_middle)), key=lambda e: dists_to_middle[e],reverse=True)[-nSamples*2:] #this works for sampling half the closest nSamples individuals
        #closest_inds=np.random.choice(closest_inds,int(len(closest_inds)/2),replace=False)
        #subsample=inds[closest_inds]
        subsample_nodes=[ts.individual(x).nodes for x in subsample] #node numbers for sampled individuals
        subsample_nodes=[a for b in subsample_nodes for a in b] #flatten the list
        subsample_nodes=np.sort(np.array(subsample_nodes))
        o=ts.simplify(subsample_nodes)
        olocs=np.array([[x.location[0],x.location[1]] for x in o.individuals()])
        if plot:
            plt.scatter(locs[:,0],locs[:,1],s=80,facecolors='none',edgecolors='k')
            plt.scatter(olocs[:,0],olocs[:,1],s=80,facecolors='r',edgecolors='r')
            plt.axis([0, np.max(locs[:,0]), 0, np.max(locs[:,1])])
    if(sampling=="point"):
        #get final generation individuals
        extant_inds=np.unique([ts.node(j).individual for j in ts.samples()]) #final-generation individuals
        extant_nodes=[ts.individual(x).nodes for x in extant_inds] #node numbers for final-generation individuals
        extant_nodes=[a for b in extant_nodes for a in b]
        ts=ts.simplify(extant_nodes)
        locs=np.array([[x.location[0],x.location[1]] for x in ts.individuals()])
        inds=np.array([x.id for x in ts.individuals()])
        mindists=[]
        for i in range(len(extant_inds)):
            row=np.array([])
            for j in range(len(sampling_locs)):
                row=np.append(row,spatial.distance.euclidean(u=locs[i],v=sampling_locs[j]))
            mindists=np.append(mindists,np.min(row))
        mindists=1/mindists**dist_scaling
        mindists=mindists/np.sum(mindists)
        subsample=np.random.choice(extant_inds,nSamples,p=mindists,replace=False)
        subsample_nodes=[ts.individual(x).nodes for x in subsample] #node numbers for sampled individuals
        subsample_nodes=[a for b in subsample_nodes for a in b] #flatten the list
        subsample_nodes=np.sort(np.array(subsample_nodes))
        o=ts.simplify(subsample_nodes)
        olocs=np.array([[x.location[0],x.location[1]] for x in o.individuals()])
        if plot:
            plt.scatter(locs[:,0], locs[:,1],s=80,facecolors='none',edgecolors='k')
            plt.scatter(olocs[:,0],olocs[:,1],s=80,facecolors='r',edgecolors='r')
            plt.axis([0, np.max(locs[:,0]), 0, np.max(locs[:,1])])
    if(write_to_file):
        o.dump(outfile)
    return o

def get_ms_outs_directory(direc,subset,start=0,stop=0):
    '''
    loop through a directory of mutated tree sequences
    and return the haplotype matrices, positions, labels,
    and spatial locations as four numpy arrays.
    If subset is true, stop and start are the
    range of files to process (0-indexed, stop not included,
    as ordered by "os.listdir(direc)").
    '''
    haps = []
    positions = []
    locs = []

    trees=[f for f in os.listdir(direc) if not f.startswith(".")]
    labels=[float(re.sub("sigma_|_.trees*|.trees","",x)) for x in trees]
    if(subset):
        trees=trees[start:stop]
        labels=labels[start:stop]
    for i in tqdm(trees):
        ts = pyslim.load(os.path.join(direc,i))
        haps.append(ts.genotype_matrix())
        positions.append(np.array([s.position for s in ts.sites()]))
        sample_inds=np.unique([ts.node(j).individual for j in ts.samples()])
        locs.append([[ts.individual(x).location[0],
               ts.individual(x).location[1]] for x in sample_inds])

    haps = np.array(haps)
    positions = np.array(positions)
    locs=np.array(locs)

    return haps,positions,labels,locs

def get_ms_outs(ts):
    '''
    get haplotypes, positions, labels, and spatial locations from a tree sequence.
    '''
    haps=np.array(ts.genotype_matrix())
    positions=np.array([s.position for s in ts.sites()])
    sample_inds=np.unique([ts.node(j).individual for j in ts.samples()])
    locs=[[ts.individual(x).location[0],
           ts.individual(x).location[1]] for x in sample_inds]
    locs=np.array(locs)

    return haps,positions,locs

def discretize_snp_positions(ogpositions):
    '''
    Takes an array of SNP positions as floats (ie from msprime) and returns
    integer positions. If two SNPs fall in the same integer position, one is
    shifted to the right by one base pair.
    '''
    count=0
    newpositions=[]
    dpos=[int(x) for x in ogpositions]
    for j in range(len(dpos)-1):
        if(dpos[j]==dpos[j+1]):
            dpos[j+1]=dpos[j+1]+1
            count=count+1
    newpositions=np.sort(dpos)
    print(str(count)+" SNPs were shifted one base pair")
    return(np.array(newpositions))

def getPairwiseIbsTractLengths(x,y,positions,maxlen,min_len_to_keep=0):
    '''
    input:
    x: haplotype as 1D array
    y: haplotype as 1D array
    positions: a 1D array of SNP positions
    maxlen: length of chromosome/contig

    Returns:
    1d array listing distances between adjacent SNPs in a pair of sequences.
    Assumes all sites are accessible.
    '''
    snps=~np.equal(x,y)
    snp_positions=positions[snps]
    l=len(snp_positions)
    ibs_tracts=[]
    if(l==0):
        ibs_tracts=[maxlen]
    else:
        if(l>1):
            ibs_tracts=snp_positions[np.arange(1,l-1,1)]-snp_positions[np.arange(0,l-2,1)] #middle blocks
        np.append(ibs_tracts,snp_positions[0]+1)          #first block
        np.append(ibs_tracts,maxlen-snp_positions[l-1]) #last block
        con=[x>=min_len_to_keep for x in ibs_tracts] #drop blocks < min_len_to_keep
        ibs_tracts=np.extract(con,ibs_tracts)
    return ibs_tracts

def getSLiMSumStats(haps,positions,label,locs,outfile,maxlen,ibs_tracts=True,verbose=True,min_len_to_keep=0):
    '''
    get summary statistics for a single SLiM tree sequence.
    '''
    if(verbose):
        print("loading genotypes")

    #load in genotypes etc into scikit-allel
    genotypes=allel.HaplotypeArray(haps).to_genotypes(ploidy=2)
    allele_counts=genotypes.count_alleles()
    genotype_allele_counts=genotypes.to_allele_counts()

    if(verbose):
         print("calculating sample-wide statistics")

    #nonspatial population-wide summary statistics
    segsites=np.shape(genotypes)[0]
    #mpd=np.mean(allel.mean_pairwise_difference(allele_counts))
    #pi=(mpd*segsites)/n_accessible_sites                               #alternate pi if not all sites are accessible
    pi=allel.sequence_diversity(positions,allele_counts,start=1,stop=1e8)
    tajD=allel.tajima_d(ac=allele_counts,start=1,stop=maxlen) #NOTE check for 0 v 1 positions
    thetaW=allel.watterson_theta(pos=positions,ac=allele_counts,start=1,stop=maxlen)
    het_o=np.mean(allel.heterozygosity_observed(genotypes))
    fis=np.mean(allel.inbreeding_coefficient(genotypes))
    sfs=allel.sfs(allele_counts[:,1])
    sfs=np.append(sfs,[np.repeat(0,np.shape(haps)[1]-len(sfs)+1)])
    #Isolation by distance
    if(verbose):
        print("calculating pairwise statistics")
    gen_dist=allel.pairwise_dxy(pos=positions,
                                gac=genotype_allele_counts,
                                start=0,
                                stop=maxlen)
    sp_dist=np.array(scipy.spatial.distance.pdist(locs))
    gen_sp_corr=np.corrcoef(gen_dist[sp_dist>0],np.log(sp_dist[sp_dist>0]))[0,1]
    print(gen_sp_corr)
    gen_dist_skew=scipy.stats.skew(gen_dist)
    gen_dist_var=np.var(gen_dist)
    gen_dist_mean=np.mean(gen_dist)

    #IBS tract length summaries
    if(verbose):
        print("calculating IBS tract lengths")
    if(ibs_tracts):
        pairs=itertools.combinations(range(np.shape(haps)[1]),2)
        ibs=[];spat_dists=[];ibs_mean_pair=[];ibs_95p_pair=[]
        ibs_var_pair=[];ibs_skew_pair=[];ibs_blocks_over_1e6_pair=[]
        ibs_blocks_pair=[]
        locs2=np.repeat(locs,2,0)
        for j in pairs:
            spdist=scipy.spatial.distance.euclidean(locs2[j[0]],locs2[j[1]])
            if spdist>0:
                ibspair=getPairwiseIbsTractLengths(haps[:,j[0]],
                                                      haps[:,j[1]],
                                                      positions,
                                                      maxlen,
                                                      min_len_to_keep)
                if len(ibspair)>0:
                    spat_dists.append(spdist)
                    ibs.append(ibspair)
                    ibs_mean_pair.append(np.mean(ibspair))
                    #ibs_95p_pair.append(np.percentile(ibspair,95))
                    #ibs_var_pair.append(np.var(ibspair))
                    ibs_skew_pair.append(scipy.stats.skew(ibspair))
                    ibs_blocks_over_1e6_pair.append(len([x for x in ibspair if x>1e6]))
                    ibs_blocks_pair.append(len(ibspair))

        #ibs stat ~ spatial distance correlations
        ibs_mean_spat_corr=np.corrcoef(ibs_mean_pair,np.log(spat_dists))[0,1] #v noisy - seems to reflect prob of sampling close relatives...
        ibs_1e6blocks_spat_corr=np.corrcoef(ibs_blocks_over_1e6_pair,np.log(spat_dists))[0,1] #better
        ibs_skew_spat_corr=np.corrcoef(ibs_skew_pair,np.log(spat_dists))[0,1] #good
        ibs_blocks_spat_corr=np.corrcoef(ibs_blocks_pair,np.log(spat_dists))[0,1] #best

        #summaries of the full distribution
        ibs_blocks_over_1e6=np.sum(ibs_blocks_over_1e6_pair)
        ibs_blocks_over_1e6_per_pair=np.mean(ibs_blocks_over_1e6_pair)
        ibs_blocks_per_pair=np.mean(ibs_blocks_pair)
        ibs_flat=np.array([x for y in ibs for x in y])
        ibs_mean=np.mean(ibs_flat)
        ibs_var=np.var(ibs_flat)
        ibs_skew=scipy.stats.skew(ibs_flat)

    if(ibs_tracts):
        if os.path.exists(outfile)==False:
            sfsnames=["sfs_"+str(x)+" " for x in range(np.shape(haps)[1])]
            sfsnames.append("sfs_"+str(np.shape(haps)[1]))
            out=open(outfile,"w")
            out.write("sigma segsites pi thetaW tajD het_o fis gen_dist_mean gen_dist_var gen_dist_skew gen_sp_corr ibs_mean ibs_var ibs_skew ibs_blocks_per_pair ibs_blocks_over_1e6_per_pair ibs_mean_spat_corr ibs_1e6blocks_spat_corr ibs_skew_spat_corr ibs_blocks_spat_corr "+
            "".join(sfsnames)+"\n")
        row=[label,segsites,pi,thetaW,tajD,het_o,fis,gen_dist_mean,gen_dist_var,gen_dist_skew,gen_sp_corr,
             ibs_mean,ibs_var,ibs_skew,ibs_blocks_per_pair,ibs_blocks_over_1e6_per_pair,ibs_mean_spat_corr,ibs_1e6blocks_spat_corr,ibs_skew_spat_corr,
             ibs_blocks_spat_corr]
        row=np.append(row,sfs)
        row=" ".join(map(str, row))
    else:
        if os.path.exists(outfile)==False:
            sfsnames=["sfs_"+str(x)+" " for x in range(np.shape(haps)[1])]
            sfsnames.append("sfs_"+str(np.shape(haps)[1]))
            out=open(outfile,"w")
            out.write("sigma segsites pi thetaW tajD het_o fis gen_dist_mean gen_dist_var gen_dist_skew gen_sp_corr"+"".join(sfsnames)+"\n")
        row=[label,segsites,pi,thetaW,tajD,het_o,fis,gen_sp_corr]
        row=np.append(row,sfs)
        row=" ".join(map(str, row))

    #append to file
    out=open(outfile,"a")
    out.write(row+"\n")
    out.close()

    return(out)
