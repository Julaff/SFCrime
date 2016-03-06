import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as pl
import warnings  # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")

sns.set(style="dark", color_codes=True)

mapdata = np.loadtxt('sf_map_copyright_openstreetmap_contributors.txt')
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

pl.imshow(mapdata, cmap = pl.get_cmap('gray'))

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sampleSub = pd.read_csv('sampleSubmission.csv')

print(train[:5])
print(test[:5])
print(sampleSub[:10])

# dataset contains 67 wrong locations, all equal to X=-120.5 Y=90 (90 is North Pole latitude!)
#train[train.Y == 90]

train['Xok'] = train[train.X != -120.5].X
train['Yok'] = train[train.Y != 90].Y

sns.jointplot(x="Xok", y="Yok", data=train)

train = train.dropna()
trainP = train[train.Category == 'PROSTITUTION'] #Grab the prostitution crimes
train = train[1:300000] #Can't use all the data and complete within 600 sec :(

#Seaborn FacetGrid, split by crime Category
g= sns.FacetGrid(train, col="Category", col_wrap=6, size=5, aspect=1/asp)

#Show the background map
for ax in g.axes:
    ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
#Kernel Density Estimate plot
g.map(sns.kdeplot, "Xok", "Yok", clip=clipsize)

pl.savefig('category_density_plot.png')

#Do a larger plot with prostitution only
pl.figure(figsize=(20,20*asp))
ax = sns.kdeplot(trainP.Xok, trainP.Yok, clip=clipsize, aspect=1/asp)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'),
              extent=lon_lat_box,
              aspect=asp)
pl.savefig('prostitution_density_plot.png')