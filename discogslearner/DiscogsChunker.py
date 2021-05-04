from typing import Any, Iterator
import xml.etree.cElementTree as ET
from gzip import GzipFile
import sys

class _Chunker:
    """
    This class cuts up large XML files into smaller chunks
    given the 'chunk_by' argument. The source needs to be a 
    URL to a gzipped xml file. This base class is made for
    the Discogs datadump releases XML, and even though it is 
    generic, it might not work for differently formatted XMLs.
    """
    def __init__(self, source: Any, chunk_by: str):
        self.__source = source
        self.__chunk_by = "</{}>".format(chunk_by)

    def chunk(self) -> Iterator:
        """
        This is the main method of the DiscogsChunker. It unpacks
        the Gzip file and returns a generator for a chunked XML.
        It returns an ElementTree object.
        """
        with GzipFile(fileobj=self.__source) as f:
            f.readline()
            data = ""
            for line in f:
                s = line.decode('UTF-8')
                data += s
                if self.__chunk_by in s:
                    yield ET.fromstring(data)
                    data = ""