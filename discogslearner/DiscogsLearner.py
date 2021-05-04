from typing import Union
import warnings
import pandas as pd
import numpy as np
import discogs_client
import math
import logging
from pandas._config.config import describe_option
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
from .DiscogsPredictions import Predictions
from .DiscogsClusters import Clusters
import sys

class Learner:
    """
    This class constructs models from extracted Discogs data. It utilizes the official Discogs client to obtain 
    Wantlist and/or Collection identifiers. 
    """
    def __init__(self, db_path: str = "", test: bool = False, token: str = "",
                 use_wantlist: bool = True, use_collection: bool = True, debug: bool = False):

        if not use_wantlist and not use_collection:
            raise Exception("Cannot construct a model without a collection or wantlist.")
        elif token == "":
            raise Exception("""annot construct a model without a personal acces token.
                            See https://www.discogs.com/settings/developers to obtain a token.""")
        
        if debug:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        logging.info("Reading Discogs DB")
        self.training_df = None
        self.__token = token
        self.__pc_df = None
        self.__releases = None
        self.__models = []
        self.__use_wantlist = use_wantlist
        self.__use_collection = use_collection
        self.__df = pd.read_csv(db_path, sep = "\t")
        if test:
            self.__df = self.__df.head(100000)

        self.__corrections()

        
    def __corrections(self) -> None:
        """
        Performs corrections before PCA. Includes scaling, but not centering
        """
        self.__df.ReleaseID = self.__df.ReleaseID.astype(int)
        self.__df.Tracks = self.__df.Tracks.astype(int)
        self.__df = self.__df.replace({"Year": {0: self.__df.Year.median()}, "Tracks": {0: self.__df.Tracks.median()}})
        self.__df.Tracks = self.__df.Tracks.clip(lower = 1, upper = 10)
        self.__df.Year = self.__df.Year.clip(lower = 1900, upper = datetime.today().year)

        self.__df = self.__df.fillna("0")
        self.__df.set_index("ReleaseID", inplace = True)
        self.__df.Year = self.__df.Year / max(self.__df.Year)
        self.__df.Tracks = self.__df.Tracks / max(self.__df.Tracks)
        self.__releases = self.__df.index
        
        self.__add_group_data("Labels")
        self.__add_group_data("Artists")
        self.__add_group_data("Companies")
        self.__df.sort_index(inplace = True)
        

    def __add_group_data(self, group: str) -> None:
        """
        Adds data about groups that should be taken into account when
        creating a PCA. Includes the size of the group and the order of
        releases. 
        """
        self.__df.sort_values([group, "Year"], inplace = True)
        sizes = self.__df[[group, "Year"]].groupby(group).size()
        self.__df["%s_size" % group] = sizes.loc[self.__df[group]].values
        self.__df["%s_sort" % group] = np.concatenate([np.arange(1, x + 1) for x in sizes]) 

        

        self.__df["%s_sort" % group] = self.__df["%s_sort" % group] / max(self.__df["%s_sort" % group])
        self.__df["%s_size" % group] = self.__df["%s_size" % group] / max(self.__df["%s_size" % group])


    def __ids_through_api(self) -> pd.Series:
        """
        Retrieves Release IDs from the wantlsit and/or collection 
        using the Discogs client
        """
        d = discogs_client.Client('Discogs Learner/0.1', user_token=self.__token)
        me = d.identity()
        collection = []
        wantlist = []
        if self.__use_collection:
            collection = [release.id for release in me.collection_folders[0].releases]
        
        if self.__use_wantlist:
            wantlist = [release.id for release in me.wantlist]
        return pd.Series(wantlist + collection)

        
    def __set_training_set(self) -> None:
        """
        Creates a subset of the database using release IDs
        """
        logging.info("Fetching IDs from Discogs")
        self.training_ids = self.__ids_through_api()
        self.training_df = self.__df.loc[self.training_ids.iloc[np.where(
            self.training_ids.isin(self.__df.index))]].drop_duplicates()
        self.training_ids = self.training_df.index
        logging.debug("Identifiers found in discogs: %s" % len(self.training_ids))

    def __format_df(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Formats a subset of the database for use in PCA.
        """
        styles = pd.get_dummies(x.Styles.str.split(";").explode()).groupby(level=0).sum()
        countries = pd.get_dummies(x.Country.explode())
        formats = pd.get_dummies(x.Formats.explode())
        df = pd.concat([styles, countries, formats, x.Year, x.Tracks], axis = 1) 
        df.columns = np.arange(len(df.columns))
        return df

    def __get_pca(self) -> tuple([PCA, list]):
        """
        This method creates a PCA transformation from the collection / wantlist.
        This allows to quantify important factors for the models.
        """
        df = self.__format_df(self.training_df)
        components = int(round(math.sqrt(len(df.columns)))) if len(df.columns) <= 15 else 15
        logging.debug("Number of PCs = %s" % components)
        pca = PCA(n_components=components)
        pca.fit(df)
        logging.debug("Variance explained: %s " % pca.explained_variance_ratio_)
        return pca, df.columns
    
    def __reduce_database(self) -> None:
        """
        This method reduces the database by only selecting Styles
        that appear in the wantlist / collection. This due to Releases
        only being interesting if their styles match.  
        """
        styles_per_release = self.__df.Styles.str.split(";").explode()
        allowed = self.training_df.Styles.str.split(";").explode().value_counts()
        
        compare = pd.concat([
            styles_per_release.iloc[np.where(styles_per_release.isin(allowed.index))].index.value_counts(),
            styles_per_release.index.value_counts()
        ], axis = 1)
        
        self.__df.set_index(self.__releases, inplace = True)
        self.__df = self.__df.iloc[np.where(compare.iloc[:,0] == compare.iloc[:,1])]
        self.__releases = self.__df.index

        self.training_df = self.__df.loc[self.training_ids[np.where(self.training_ids.isin(self.__df.index))]].drop_duplicates()
        self.training_ids = self.training_df.index
        logging.debug("Identifiers found after reduction: %s" % len(self.training_ids))

        
    def __transform_to_pc_space(self, x, pca: PCA, columns: list) -> pd.DataFrame:
        """
        Transforms a dataframe to PCA space. This is used for the training
        data, but also for the discogs database.
        """
        sub = self.__format_df(x)
        for c in columns:
            if c not in sub.columns:
                sub[c] = 0
        return pd.DataFrame(pca.transform(sub[columns]), index = sub.index)


    def __get_cluster_df(self) -> pd.DataFrame:
        """
        This method returns the mean and variance of a given group and maps
        those to individual releases. 
        """
        groups = []
        columns = ["Artists", "Labels", "Companies"]
        for i, group in enumerate(columns):
            tqdm.pandas(desc = "Creating %s map (%s/%s)" % (group, i + 1, len(columns)), total = len(self.__df[group].unique()))
            cluster_data = self.__pc_df.groupby(self.__df[group]).progress_apply(lambda x: x.mean()).apply(pd.to_numeric, downcast="float").round(3)
            cluster_data.columns = np.arange(0, len(cluster_data.columns)) 
            means = cluster_data.loc[self.__df[group]].set_index(self.__pc_df.index)
            groups.append(pd.concat([means, self.__pc_df.subtract(means) ** 2], axis = 1))
        return pd.concat(groups, axis = 1)

    def __adjust_pc_df(self, pca: PCA, columns: list) -> None:
        """
        This method maps the Discogs database to PCA space and adds 
        the mean and variance of groups to the PCA dataframe. To save RAM,
        this is done in batches.
        """
        self.__df.reset_index(inplace = True) 
        tqdm.pandas(desc = "Mapping database to PCA space", total = len(self.__df.index) // 5000)
        self.__pc_df = self.__df.groupby(self.__df.index // 5000).progress_apply(
            lambda x: self.__transform_to_pc_space(x, pca, columns)
        ).reset_index(drop=True).apply(pd.to_numeric, downcast="float").round(3)
        self.__df.set_index("ReleaseID", inplace = True)
        self.__df = self.__df[["Labels", "Artists", "Companies"]]
        self.__pc_df.set_index(self.__df.index, inplace = True)
        self.__pc_df = pd.concat([self.__pc_df, self.__get_cluster_df()], axis = 1)
        self.__pc_df.columns = np.arange(0, len(self.__pc_df.columns))
        logging.debug("Final PC DataFrame:\n%s" % self.__pc_df.head())


    def __create_train_test(self, X1: pd.DataFrame) -> list:
        """
        This method creates a train-test split using 
        the training identifiers determined from the 
        collection / wantlist.
        """
        
        X1["Pred"] = 1
        X2 = self.__pc_df.sample(len(X1.index))
        X2["Pred"] = 0
        
        X = pd.concat([X1, X2])
        labels = X.Pred
        X.drop("Pred", inplace = True, axis = 1)
        return train_test_split(X, labels, test_size=0.3)
        
    def __create_models(self, n_models: int, data: pd.DataFrame) -> None:
        """
        Creates N models. Uses a Random Forest classifier with 
        5-fold cross-validation to prevent overfitting.
        """
        scores = []
        models = []
        with tqdm(desc = "Creating models", total=n_models) as pbar:
            for _ in range(n_models):
                X_train, X_test, y_train, y_test = self.__create_train_test(data)
                clf = RandomForestClassifier(n_estimators = 100)
                cv_results = cross_validate(clf, X_train, y_train, 
                                            cv=5, scoring = "accuracy", 
                                            return_estimator = True)
                model = cv_results["estimator"][np.argmax(cv_results["test_score"])]
                scores.append(accuracy_score(y_test, model.predict(X_test)))
                self.__models.append(model)
                pbar.update(1)
        score = np.mean(scores)
        logging.debug("Model accuracy: %s" % score)
        return score

    def learn_and_predict(self, n_models: int=10) -> list:
        """
        The main method to be called for training models and 
        predicting Releases. Uses 10 models by default, should
        use less than 5 models.
        Returns a sorted Pandas Series object with Release IDs 
        as index and probabilities as values. 
        """
        self.__set_training_set()
        self.__reduce_database()
        pca, columns = self.__get_pca()
        self.__adjust_pc_df(pca, columns)
        self.__create_models(n_models, data = self.__pc_df.loc[self.training_ids])
        return self.__predict()      

    def __predict(self) -> Predictions:
        """
        Predict similarity of releases to the trained models. 
        Requires 'start_learning' to be called first. 
        Returns a sorted Pandas Series object with Release IDs 
        as index and probabilities as values.  
        """
        total = []
        with tqdm(desc = "Predicting Releases", total=len(self.__models)) as pbar:
            for model in self.__models:
                total.append(pd.DataFrame(model.predict_proba(self.__pc_df), index=self.__pc_df.index)[1])
                pbar.update(1)
            preds = pd.concat(total, axis = 1).prod(axis=1).sort_values(ascending=False)
            preds = preds.iloc[np.where(preds.index.isin(self.training_ids) == False)].to_frame()
        return Predictions(
            df = self.__df,
            pc_df = self.__pc_df,
            training_ids= self.training_ids,
            predictions=preds
        )

    
    
    
