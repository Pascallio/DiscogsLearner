import json
import requests
import re
import pandas as pd
import numpy as np
from .DiscogsChunker import _Chunker as Chunker
from datetime import datetime
from urllib.request import urlopen, Request
from tqdm import tqdm
import xml.etree.cElementTree as ET
import os

class Extracter:
    """
    This class extracts data from the monthly Discogs datadumps.
    By default it selects the recent Releases, but a custom URL
    can be given if needed.
    """
    def __init__(self, genre: str="Electronic", url: str = None):
        today = datetime.today()
        self.__url = url
        if url is None:
            self.__url = "https://discogs-data-dumps.s3-us-west-2.amazonaws.com/data/%d/discogs_%d%02d01_releases.xml.gz" % (today.year, today.year, today.month)     

        self.__main = []
        self.__genre = genre
        self.__mapping_styles = {}
        self.__mapping_formats = {}
        self.__mapping_countries = {}
        self.reduced = False
        self.__n_releases = self.__get_n_releases()
        self.__output = None
    
    def __get_n_releases(self) -> int:
        """
        Retrieves the number of releases in the database to use as indicator.
        """
        url = "https://api.discogs.com/"
        response = json.loads(requests.get(url).text)
        return int(response["statistics"]["releases"])
            
    def extract(self, output: str) -> None:
        """
        Main method for extracting Discogs data. Uses the Chunker class to
        cut up the XML and save RAM. Needs an output file for the data
        to be stored. Has to be run only once per month to keep updated.
        """
        if output is None:
            raise Exception("No output file given.")

        
        os.makedirs(os.path.dirname(output), exist_ok=True)
        file = os.path.join(output, "releases.tsv")

        a = open(file, "w")
        a.write("\t".join(["ReleaseID", "Labels", "Format", "Year", "Country",
                            "Artists",  "Styles", "Tracks", "Companies"]) + "\n")
        a.close()
        
        with urlopen(Request(self.__url, headers={"Accept-Encoding": "gzip"})) as response:
            chunker = Chunker(response, chunk_by = "release")
            with tqdm(desc = "Parsing releases from Discogs", total=self.__n_releases) as pbar:
                self.extractChunks(chunker, pbar, file)

        self.createOneHotFiles(output)

    def extractChunks(self, chunker, pbar, file):
        for i, chunk in enumerate(chunker.chunk()):

            self.__extract_chunk_reduced(chunk)
            pbar.update(1)

            if i % 100000 == 0:
                df = pd.DataFrame(self.__main)
                df.to_csv(
                    file, 
                    mode = "a", 
                    header = False, 
                    sep = "\t", 
                    index = False
                )

                self.__main = []

        df = pd.DataFrame(self.__main)
        df.to_csv(
            file, 
            mode = "a", 
            header = False, 
            sep = "\t", 
            index = False
        )

        self.__main = []

    def createOneHotFiles(self, output):
        file = os.path.join(output, "releases.tsv")

        idx = self.getIndexes(file, minimum_values = 250_000)


        self.create_dummies(
            file = file, 
            output = os.path.join(output, "countries.tsv"), 
            column = "Country", 
            mask = idx, 
            explode = False
        )
        self.create_dummies(
            file = file, 
            output = os.path.join(output, "formats.tsv"), 
            column = "Formats", 
            mask = idx, 
            explode = False
        )
        self.create_dummies(
            file = file, 
            output = os.path.join(output, "styles.tsv"), 
            column = "Styles", 
            mask = idx, 
            explode = True
        )

    def __get_mapping(self, mapping: dict, x: str) -> int:
        """
        Converts a new value to a integer, saving storage space.
        """
        return mapping.setdefault(x, len(mapping))

    def __extract_chunk(self, chunk: ET) -> None:
        if chunk.findtext(".//genre", "Unknown") == self.__genre:
            id = chunk.attrib["id"]
            year = int(re.sub(r'[^0-9]+', '0', chunk.findtext("released", "0"))[:4])
            country = chunk.findtext("country", "Unknown")
            tracks = len(chunk.findall("tracklist//track//title"))
            format = chunk.find("formats//format").attrib["name"]
            styles = ";".join({x.text for x in chunk.findall(".//style")})

            labels = {re.sub(r'[^0-9]+', "0", l.attrib.get("id", "0")) for l in chunk.findall(".//label")}            
            artists = {x[0].text for x in chunk.findall(".//artist")}
            company = {x[0].text for x in chunk.findall("companies//company")}
            self.__main.append((id, labels, format, year, country, artists, styles, tracks, company))

        
    def __extract_chunk_reduced(self, chunk: ET) -> None:
        """
        Finds values in a Release and adds them to a list. 
        """
        if chunk.findtext(".//genre", "Unknown") == self.__genre:
            id = chunk.attrib["id"]
            released = int(re.sub(r'[^0-9]+', '0', chunk.findtext("released", "0"))[:4])
            tracks = len(chunk.findall("tracklist//track//title"))

            country = self.__get_mapping(self.__mapping_countries, chunk.findtext("country", "Unknown"))
            styles = ";".join({str(self.__get_mapping(self.__mapping_styles, x.text)) for x in chunk.findall(".//style")})
            format = self.__get_mapping(self.__mapping_formats, chunk.find("formats//format").attrib["name"])

            artists = ";".join({x[0].text for x in chunk.findall(".//artist")})
            labels = ";".join({re.sub(r'[^0-9]+', "0", l.attrib.get("id", "0")) for l in chunk.findall(".//label")})
            company = ";".join({x[0].text for x in chunk.findall("companies//company")}) 
            self.__main.append((id, labels, format, released, country, artists, styles, tracks, company))


    def getIndexes(self, path, minimum_values = 10000):
        
        data = pd.read_csv(path, sep = "\t")["Country"]
        counts = data.value_counts()
        mask = data.isin(counts.index[counts > minimum_values])
        index1 = data.loc[mask].index

        data = pd.read_csv(path, sep = "\t")["Styles"].str.split(";").explode()
        counts = data.value_counts()
        mask = data.isin(counts.index[counts > minimum_values])
        index2 = data.loc[mask].index
        return np.intersect1d(index1, index2)

    def create_dummies(self, file, output, column, mask, explode = False):

        from tqdm import tqdm

        tqdm.pandas(desc = "Mapping database to PCA space", total = len(data.index) // 5000)
        data = pd.read_csv(file, sep = "\t")[column].loc[mask]
        data.groupby(data.index // 5000).progress_apply(
            lambda x: pd.get_dummies(x["Styles"].str.split(";").explode()).groupby(level = 0).sum().to_csv("Data/test2.tsv", mode = "a",   sep = "\t")
        )

        
        if explode:
            data = data.str.split(";").explode()

        dummies = pd.get_dummies(data).apply(pd.to_numeric, downcast="integer")

        a = open(output, "w")
        a.write("\t".join(list(dummies.columns.astype(str))) + "\n")
        a.close()

        if explode:
            import dask.dataframe as dd
            dummies["id"] = dummies.index
            dummies = dd.from_pandas(dummies, chunksize=10000)
            dummies = dummies.groupby('id').sum().reset_index().compute().drop(columns = "id")

        dummies.to_csv(
            output, 
            mode = "a", 
            header = False, 
            sep = "\t", 
            index = True
        )   

