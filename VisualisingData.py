import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from Experiments import getXandY
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def visualisingVectors(X,Y):
	df = pd.DataFrame()

	pca_50 = PCA(n_components=50)
	pca_result_50 = pca_50.fit_transform(X)
	tsne = TSNE(n_components=2,perplexity=50, n_iter=5000)
	tsne_pca_results = tsne.fit_transform(X)
	df["tsne-pca50-one"] = tsne_pca_results[: ,0]
	df['tsne-pca50-two'] = tsne_pca_results[: ,1]
	df['Y'] = Y
	plt.figure()
	sns.scatterplot(
		x="tsne-pca50-one", y="tsne-pca50-two",
		hue="Y",
		palette=sns.color_palette("hls", 2),
		data=df,
		legend="full",
	)
	plt.title = "Lemma"
	plt.show()
X, Y = getXandY(filePath="./vectors/mpnet/mpnet-stopWords.csv")
visualisingVectors(X,Y)
