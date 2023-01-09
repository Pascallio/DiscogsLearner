import numpy as np
import os
import pandas as pd
import igraph as ig

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn import preprocessing
from itertools import combinations

class DiscogsTransformer:
    def __init__(self):
        pass

    def __getColumns(self, series, percentage):
        vals = series.astype(str).str.split(";").explode()
        counts = vals.value_counts() 
        relative = counts * 100 / sum(counts)
        id = min(np.where(np.floor(np.cumsum(relative) >= percentage))[0])
        return counts.index[0:id]

    def __parse(self, x, output, max_value, columns):
        x = pd.get_dummies(x.astype(str).str.split(";").explode())
        x = x.groupby(level = 0).sum().fillna(0).drop(columns = 'nan', errors = "ignore")
        
        all = pd.Series(columns).astype(str)
        cols = list(all.loc[~all.isin(x.columns)])
        remained = pd.DataFrame(0, index = x.index, columns=cols)
        x = pd.concat([x, remained], axis = 1)[columns]
        x = x.iloc[np.where(x.sum(axis = 1) > 0)]
        x.to_csv(output, mode = "a", sep = "\t", header = False)

    def __parseColumn(self, data, category, cols, folder, groupSize = 10_000):
        columns = cols[category]
        groups = data.index // groupSize
        tqdm.pandas(desc = "Parsing column " + category, total = len(data.index) // groupSize)

        max_value = int(data[category].groupby(groups).apply(
            lambda x: x.astype(str).str.split(";").explode().astype(float).max()
        ).max())
        
        output = os.path.join(folder, category + ".tsv")
        header = pd.DataFrame(columns = columns)
        header.to_csv(output, mode = "w", sep = "\t")
        data.groupby(groups).progress_apply(
            lambda x: self.__parse(x[category], output, max_value, columns)
        )

    def transform(self, data, columns):

        cols = {_: self.__getColumns(data[_], 90) for _ in columns}
        for column in columns:
            self.__parseColumn(data, column, cols, "Data")


        idx = set.intersection(
            set(pd.read_csv("Data/Styles.tsv", index_col=0, sep = "\t").index),
            set(pd.read_csv("Data/Country.tsv", index_col=0, sep = "\t").index),
            set(pd.read_csv("Data/Formats.tsv", index_col=0, sep = "\t").index)
        )

        df = pd.concat([
            pd.read_csv("Data/Styles.tsv", index_col=0, sep = "\t").loc[idx],
            pd.read_csv("Data/Country.tsv", index_col=0, sep = "\t").loc[idx],
            pd.read_csv("Data/Formats.tsv", index_col=0, sep = "\t").loc[idx],
            pd.read_csv("Data/releases.tsv", sep = "\t", usecols = ["Year", "Tracks"]).loc[idx]
        ], axis = 1)

        df = df.replace({"Year": {0: df.Year.median()}, "Tracks": {0: df.Tracks.median()}})
        df.Tracks = df.Tracks.clip(lower = 1, upper = 10)
        df.Year = df.Year.clip(lower = 1900, upper = 2022)
        pca = PCA(n_components=20)
        df = preprocessing.StandardScaler().fit_transform(df)

        pd.DataFrame(pca.fit_transform(df).round(4)).set_index(pd.Index(idx)).to_csv("Data/Release_PCA.tsv", sep = "\t")



    def transformGroups(self):
        df = pd.read_csv("Data/Release_PCA.tsv", index_col=0, sep = "\t")

        for column in ["Labels", "Companies", "Artists"]: 
            file = "Data/releases.tsv"
            data = pd.read_csv(file, sep = "\t")[column].loc[df.index]
            data = data.str.split(";").explode().dropna().astype(int)

            tqdm.pandas(desc = "Parsing " + column, total = len(data.value_counts().index))

            translation = data.groupby(data).progress_apply(
                lambda x: pd.DataFrame(df.loc[x.index].mean(axis = 0)).transpose()
            ).droplevel(1)
            translation.to_csv(f"Data/{column}_PCA.tsv", sep = "\t")

            tqdm.pandas(desc = "Merging into releases " + column, total = len(data.index))
            data.groupby(level = 0).progress_apply(
                lambda x:  pd.DataFrame(translation.loc[x].mean(axis = 0)).transpose()
            ).droplevel(1).to_csv(f"Data/{column}_release_PCA.tsv", sep = "\t")


    def transformNetwork(self):
        df = pd.Series(data = [0, 1, 2, 3, 1], index = [1,1,1,2,2])
        edges = pd.DataFrame(df.groupby(level=0).apply(
            lambda x: list(combinations(x, 2))
        ).explode().to_list(), columns = ["from", "to"])

        g = ig.Graph.DataFrame(edges, directed = False)
        g.get_vertex_dataframe()
        ig.plot(g)