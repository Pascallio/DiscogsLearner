from setuptools import setup

setup(
   name='discogslearner',
   version='0.1',
   description='Machine Learning module for Discogs',
   author='Pascal Maas',
   author_email='p.maas92@gmail.com',
   packages=['discogslearner'],
   install_requires=['pandas', 'tqdm', 'numpy', "sklearn", "discogs_client"]
)