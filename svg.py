import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
#import SpaGFT as spg
from scipy.sparse import csr_matrix
import anndata as ad
import contextlib
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, ConfusionMatrixDisplay
from scipy.stats import spearmanr, pearsonr
# import squidpy as sq


_Metric = [
    'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
]


def calcuate_spatial_neighbours(
    adata: ad.AnnData, 
    knn: bool = True, 
    n_neighbours: int = 8, 
    metric: str = 'euclidean', 
    method='umap',
    inplace=True):

    assert metric in _Metric, "Give a valid metric from this set of metrics: 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'."
    # call scanpy's neighbors function on spatial representation
    sc.pp.neighbors(adata, use_rep='spatial', n_neighbors=n_neighbours, knn=knn, method=method, metric=metric)
    # if method is Gaussian Kernel, we need to convert connectivities and distances to sparse matrices
    if method=='gauss':
        adata.obsp['connectivities'] = csr_matrix(adata.obsp['connectivities'])
        adata.obsp['distances'] = csr_matrix(adata.obsp['distances'])

    return None if inplace else adata


def calculate_morans_i_and_ranks(
    adata: ad.AnnData, 
    percentile: float =10.0):

    # check if neighbours are calculated
    assert hasattr(adata, "obsp") and "connectivities" in adata.obsp, "Neighbours must be calculated, reffer to calcuate_spatial_neighbours function!"
    # calculate global Moran's I and save results
    results = sc.metrics.morans_i(adata)
    adata.var['morans'] = results
    #gene_ranks = adata.var['morans'].abs().rank(ascending=False).sort_index() 
    gene_ranks = adata.var['morans'].rank(ascending=False).sort_index() 
    rnk_threshold = np.percentile(gene_ranks, percentile)
    gene_svgs = gene_ranks[gene_ranks<rnk_threshold].sort_values().index

    return results, gene_ranks, gene_svgs

def calculate_gearys_c_and_ranks(
    adata: ad.AnnData, 
    percentile: float =10.0):

    # check if neighbours are calculated
    assert hasattr(adata, "obsp") and "connectivities" in adata.obsp, "Neighbours must be calculated, reffer to calcuate_spatial_neighbours function!"
    # calculate global Moran's I and save results
    results = sc.metrics.gearys_c(adata)
    adata.var['gearys'] = results
    gene_ranks = (adata.var['gearys']-1).abs().rank(ascending=False).sort_index()
    rnk_threshold = np.percentile(gene_ranks, percentile)
    gene_svgs = gene_ranks[gene_ranks<rnk_threshold].sort_values().index

    return results, gene_ranks, gene_svgs

def combine_results(ranks, weights):
    # combine ranks obtain with different metrics 
    gene_ranks_combined = np.zeros(ranks[0].shape)
    for i, r in enumerate(ranks):
        gene_ranks_combined += weights[i]*r      
    gene_ranks_combined = gene_ranks_combined.rank(ascending=True).sort_index()
    
    return gene_ranks_combined

def calculate_percent_by_annotation(
    adata: ad.AnnData, 
    genes=[]):
 
    if len(genes)==0: genes = adata.var_names
    annotation_gene_matrix = []
    for gene in genes:
            dist = adata.obs.iloc[adata[:, gene].X.nonzero()[0]].annotation.value_counts(sort=False, normalize=True).sort_index().values
            annotation_gene_matrix.append(dist)

    return annotation_gene_matrix

def calculate_weight_matrix(adata: ad.AnnData, 
                            weights=[1.5, 0.5]):
    df = df = pd.DataFrame(adata.obs.annotation)
    N = df.shape[0]

    annotations = df['annotation'].values

    annotations_2d = np.array([annotations]*N)

    adj_matrix = np.where(annotations_2d == annotations_2d.T, weights[0], weights[1])
    
    connectivities = adata.obsp['connectivities']
    connectivities_new = csr_matrix(adj_matrix).multiply(connectivities)
    
    return connectivities_new


def calculate_num_cells_by_annotation(
    adata: ad.AnnData, 
    genes=[]):
     
    if len(genes)==0: genes = adata.var_names
    annotation_gene_matrix = []
    for gene in genes:
            dist = adata.obs.iloc[adata[:, gene].X.nonzero()[0]].annotation.value_counts(sort=False).sort_index().values
            annotation_gene_matrix.append(dist)

    return annotation_gene_matrix

def spa_gft(
    adata: ad.AnnData, 
    ratio_neighbors=1, 
    spatial_info='spatial', 
    filter_peaks=True,
    S=6,
    qvalue_cutoff=0.05):
    # steps are obtained from this tutorial: https://spagft.readthedocs.io/en/latest/spatial/codex_A6.html
    # determine the number of low-frequency FMs and high-frequency FMs
    ratio_low, ratio_high = spg.gft.determine_frequency_ratio(adata, ratio_neighbors=ratio_neighbors, spatial_info=spatial_info)
    # calculation
    gene_df = spg.detect_svg(adata,
                             spatial_info='spatial',
                             ratio_low_freq=ratio_low,
                             ratio_high_freq=ratio_high,
                             ratio_neighbors=ratio_neighbors,
                             filter_peaks=filter_peaks,
                             S=S)
    # extract spaitally variable genes; use qvalue as cutoff to obtain spatial variable features
    svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < qvalue_cutoff].index.tolist()
    svg_ranks = gene_df.sort_index().svg_rank
    return gene_df, svg_list, svg_ranks

def report(
    adata: ad.AnnData, 
    spa_gft_svgs, 
    predicted_ranks, 
    draw_cm=True):
    
    predicted_svgs = predicted_ranks.sort_values()[0:len(spa_gft_svgs)]
    labels_truth = [gene in spa_gft_svgs for gene in adata.var_names]
    labels_pred = [gene in predicted_svgs for gene in adata.var_names]

    cm = confusion_matrix(labels_truth, labels_pred)
    f1 = f1_score(labels_truth, labels_pred)
    auc = roc_auc_score(labels_truth, labels_pred)
    if draw_cm:
        display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        display.plot()
        plt.show()
    return cm, f1, auc

def report1(
    adata: ad.AnnData, 
    spa_gft_svgs, 
    predicted_svgs, 
    draw_cm=True):
    
    labels_truth = [gene in spa_gft_svgs for gene in adata.var_names]
    labels_pred = [gene in predicted_svgs for gene in adata.var_names]

    cm = confusion_matrix(labels_truth, labels_pred)
    f1 = f1_score(labels_truth, labels_pred)
    auc = roc_auc_score(labels_truth, labels_pred)
    if draw_cm:
        display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
        display.plot()
        plt.show()
    return cm, f1, auc
                         
def rank_correlation(
    spagft_ranks, 
    predicted_ranks, 
    corr='spearman', 
    genes=[]):
    
    if genes:
        spagft = spagft_ranks[genes].values
        predicted = predicted_ranks[genes].values
    else:
        spagft = spagft_ranks.values
        predicted = predicted_ranks.values
        
    if corr=='spearman':
        correlation, p_value = spearmanr(spagft, predicted)
    elif corr=='pearson':
        correlation, p_value = pearsonr(spagft, predicted)
        
    return correlation, p_value

def most_different_genes(
    spagft_ranks, 
    predicted_ranks,
    count=10):
    
    most_defferent = abs(spagft_ranks - predicted_ranks).sort_values(ascending=False)[0:count].index
    
    return most_defferent

##### utility functions #####

def delete_module(module_name: str, paranoid=None):
    from sys import modules
    try:
        this_module = modules[module_name]
    except KeyError as e:
        raise ValueError(module_name) from e
    these_symbols = dir(this_module)
    if paranoid:
        try:
            paranoid[:]  # sequence support
        except:
            raise ValueError('must supply a finite list for paranoid')
        else:
            these_symbols = paranoid[:]
    del modules[module_name]
    for mod in modules.values():
        with contextlib.suppress(AttributeError):
            delattr(mod, module_name)
        if paranoid:
            for symbol in these_symbols:
                if symbol[:2] == '__':  # ignore special symbols
                    continue
                with contextlib.suppress(AttributeError):
                    delattr(mod, module_name)
                    

def open_file(path):
    with open(path) as f:
        svg_spagft_results = pickle.load(f)
    return svg_spagft_results

def save_file(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)
        
def visualize(adata, gene_list):
    x = adata.obsm['spatial'][:,0]
    y = adata.obsm['spatial'][:,1]
    adata.obs['x'] = x.astype(int)
    adata.obs['y'] = y.astype(int)
    
    fig, axes = plt.subplots(nrows=len(gene_list)//5, ncols=5, figsize=(12, (len(gene_list)//5)*3))
    axes = axes.flatten()

    # Iterate over the subplots
    for i, g in enumerate(gene_list):
        # Get x and y values for the current subplot
        gen = adata[:,adata.var.index == g]
        x = gen.obs['x'].values
        y = -gen.obs['y'].values

        # Create scatter plot on the current subplot
        scatter = axes[i].scatter(x, y, c=gen.X.A, cmap='jet', s=1 )

        # Set subplot title
        axes[i].set_title(f'Gene {g}')

        # Remove ticks
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Add c9olorbar
        colorbar = plt.colorbar(scatter, ax=axes[i], pad=0.02, fraction=0.05, aspect=15)

    # Adjust the spacing between subplots
    plt.tight_layout()


    # Display the plot
    plt.show()
    
def visualize_scanpy(adata, gene_list):
    # Assuming the spatial coordinates are stored in the 'spatial' attribute
    # Adjust the values according to your dataset's actual attributes
    spatial_coords = adata.obsm['spatial']

    # Create a spatial scatter plot
    sc.pl.spatial(adata, color=gene_list, spot_size=1, wspace=0)
    