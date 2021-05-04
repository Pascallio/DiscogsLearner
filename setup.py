from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version = long_description.split("Version: ")[1][0:3]

setup(
   name='discogslearner',
   version=version,
   license = "GPL-3.0",
   description='Machine Learning module for Discogs',
   author='Pascal Maas',
   author_email='p.maas92@gmail.com',
   packages=['discogslearner'],
   install_requires=['pandas', 'tqdm', 'numpy', "sklearn", "discogs_client"],
   url = "https://github.com/Pascallio",
   download_url = "https://github.com/Pascallio/DiscogsLearner/archive/refs/tags/v0.2.tar.gz",
   keywords = ["Discogs", "Machine Learning"],
   long_description=long_description,
   long_description_content_type='text/markdown'
)