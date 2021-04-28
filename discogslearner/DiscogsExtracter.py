import json
import requests
import re
import pandas as pd
from .DiscogsChunker import _Chunker as Chunker
from datetime import datetime
from urllib.request import urlopen, Request
from tqdm import tqdm

class Extracter:
    """
    This class extracts data from the monthly Discogs datadumps.
    By default it selects the recent Releases, but a custom URL
    can be given if needed.
    """
    def __init__(self, genre="Electronic", url = None):
        today = datetime.today()
        self.__url = url
        if url is None:
            self.__url = "https://discogs-data.s3-us-west-2.amazonaws.com/data/%d/discogs_%d%02d01_releases.xml.gz" % (today.year, today.year, today.month)            

        self.__main = []
        self.__genre = genre
        self.__mapping_styles = {}
        self.__mapping_formats = {}
        self.__mapping_countries = {}
        self.__n_releases = self.__get_n_releases()
    
    def __get_n_releases(self):
        """
        Retrieves the number of releases in the database to use as indicator.
        """
        url = "https://api.discogs.com/"
        response = json.loads(requests.get(url).text)
        return int(response["statistics"]["releases"])
            
    def extract(self, output):
        """
        Main method for extracting Discogs data. Uses the Chunker class to
        cut up the XML and save RAM. Needs an output file for the data
        to be stored. Has to be run only once per month to keep updated.
        """
        if output is None:
            raise Exception("No output file given.")
        a = open(output, "w")
        a.write("\t".join(["ReleaseID", "Labels", "Formats", "Year", "Country","Artists", 
                               "Styles", "Tracks", "Companies"]) + "\n")
        a.close()

        with urlopen(Request(self.__url, headers={"Accept-Encoding": "gzip"})) as response:
            c = Chunker(response, chunk_by = "release")
            with tqdm(desc = "Parsing releases from Discogs", total=self.__n_releases) as pbar:
                for i, chunk in enumerate(c.chunk()):
                    self.__extract_chunk(chunk)
                    if i % 10000 == 0:
                        self.__write(pd.DataFrame(self.__main), output)
                    pbar.update(1)
            self.__write(pd.DataFrame(self.__main), output)

    def __get_mapping(self, mapping, x):
        """
        Converts a new value to a integer, saving storage space.
        """
        return mapping.setdefault(x, len(mapping))
        
    def __extract_chunk(self, chunk):
        """
        Finds values in a Release and adds them to a list. 
        """
        if chunk.findtext(".//genre", "Unknown") == self.__genre:
            id = chunk.attrib["id"]
            released = int(re.sub(r'[^0-9]+', '0', chunk.findtext("released", "0"))[:4])
            labels = ";".join({re.sub(r'[^0-9]+', "0", l.attrib.get("id", "0")) for l in chunk.findall(".//label")})
            country = self.__get_mapping(self.__mapping_countries, chunk.findtext("country", "Unknown"))
            styles = ";".join({str(self.__get_mapping(self.__mapping_styles, x.text)) for x in chunk.findall(".//style")})
            artists = ";".join({x[0].text for x in chunk.findall(".//artist")})
            tracks = len(chunk.findall("tracklist//track//title"))
            format = self.__get_mapping(self.__mapping_formats, chunk.find("formats//format").attrib["name"])
            company = ";".join({x[0].text for x in chunk.findall("companies//company")}) 
            self.__main.append((id, labels, format, released, country, artists, styles, tracks, company))

    def __write(self, df, target):
        """
        Writes processed chunks to the given file.
        """
        df.to_csv(target, mode = "a", header = False, sep = "\t", index = False)
        self.__main = []