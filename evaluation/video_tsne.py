import numpy as np
import pandas as pd 
import torch 
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Credit:
# https://github.com/sinanatra/image-tsne/blob/master/notebooks/image_tsne.ipynb
# https://www.kaggle.com/code/dehaozhang/t-sne-visualization
def gen_tsne(model, test_loader, device, plot_path):
    model.to(device)
    model.eval()

    # Generate video embeddings 
    embeddings, labels = [], []
    for imgs, label, _ in test_loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            # imgs = Variable(imgs)
            emb = model.get_embedding(imgs)	

        embeddings.extend(emb.cpu().numpy())
        labels.extend(label)

    # PCA transform
    embeddings = np.array(embeddings)
    pca = PCA(n_components=300)
    pca.fit(embeddings)
    pca_features = pca.transform(embeddings)

    # Train a t-SNE
    X = np.array(pca_features)
    tsne = TSNE(n_components=2, learning_rate=350, 
                perplexity=30, angle=0.2, verbose=2).fit_transform(X)

    # Plot the clusters
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    df_cluster = pd.DataFrame({'x': tx, 'y': ty, 'label': labels})
    n_cls = len(set(labels))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(data=df_cluster, x='x', y='y', hue='label', 
                    palette=sns.hls_palette(n_cls), 
                    legend='full');
    plt.savefig(plot_path)