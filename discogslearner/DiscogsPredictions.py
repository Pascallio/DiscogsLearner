import pandas as pd
import numpy as np
import pickle

class Predictions:
    """
    This class allows for post-processing predictions made by the algorithm.
    It allows for saving and loading previous predictions. Releases can be 
    filtered by unseen groups, like Artists and/or Labels.
    """
    def __init__(self, df: pd.DataFrame = None, pc_df: pd.DataFrame = None, 
                training_ids: pd.Series = None, predictions: pd.DataFrame = None):
        self.__df = df
        self.__pc_df = pc_df
        self.__training_ids = training_ids
        self.__predictions = predictions

    def get_top_n(self, n: int=10):
        """
        Returns the N Releases with the highest probability
        """
        return self.__predictions.iloc[:n]
    

    def filter_group(self, group: str) -> pd.Series:
        """
        This method filters out predictions in groups alread present
        in the Wantlist / Collection. This feature can be used to discover
        similar releases on unseen Labels, Artists, or Companies.
        """
        if group not in self.__df.columns:
            raise Exception("""Group not found. Must be one of the following:
            Artists, Labels, Companies""")

        allowed_indexes = self.__df.index[~self.__df[group].isin(self.__df.loc[self.__training_ids, group])]
        mask = np.where(self.__predictions.index.isin(allowed_indexes))
        return self.__predictions.iloc[mask]

    
        

    def load(self, output: str) -> None:
        """
        Loads previously saved predictions.
        """
        with open(output, 'rb') as file:
            tmp_dict = pickle.load(file)     

        self.__dict__.update(tmp_dict) 


    def save(self, output: str) -> None:
        """
        Save the current predictions made.
        """
        with open(output, 'wb') as file:
            pickle.dump(self.__dict__, file, 2)


    def __repr__(self) -> str:
        """
        Prints the top 10 most similar releases as default.
        """
        return "Top 10 most similar Releases:\n{0}".format(self.__predictions.head(10))