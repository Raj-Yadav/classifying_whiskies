
import numpy as np
import pandas as pd

whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
whisky.columns

flavors = whisky.iloc[:, 2:14]
flavors.columns
flavors.head(2)

## computing and ploting correlation matrix of flavor dataframe
corr_flavor = pd.DataFrame.corr(flavors)

import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
plt.pcolor(corr_flavor)
plt.colorbar()
plt.savefig("corr_flavor.pdf")

xm = flavors.transpose()
corr_whisky = pd.DataFrame.corr(xm)

plt.figure(figsize = (10,10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.axis("tight")
plt.savefig("corr_whisky.pdf")

from sklearn.cluster.bicluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whisky)

xx = np.sum(model.rows_, axis = 1)

##########################
 





