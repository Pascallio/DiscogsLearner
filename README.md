# DiscogsLearner - ML library for Discogs

<!--- Version: 0.2 ---> 

## Introduction
This package enables predicting similar releases using your Discogs Wantlist and/or Collection. To accomplish this, a 2-step process is executed: Data retrieval using the monthly data dumps and data learning using a list of identifiers obtained from your Wantlist and/or Collection. It produces release identifiers together with probabilities of similarity to your input. See *Details* for an in-depth explanation. This package requires about 3GB of free RAM to process the whole 'Electronic' genre.

## Installation

    pip install discogslearner
   
## Usage 

1. Obtain a Discogs personal access token. See https://www.discogs.com/settings/developers on how to obtain one.
2. Execute a script like the following:

```python
import discogslearner

if __name__ == "__main__":
    output_file = "Data/discogs_db.tsv"
    my_genre = "Electronic"
    my_token = "your_token_here"
    
    extracter = discogslearner.Extracter(genre = my_genre)
    extracter.extract(output = output_file)
    learner = discogslearner.Learner(db_path = output_file, 
                                    use_wantlist=True, 
                                    use_collection=True,
                                    token = my_token)

    outcome = learner.learn_and_predict(n_models = 10)
    print(outcome)
```

## Details
In order to learn from Discogs data, the fields Format, Year, Country, Style(s) and Number of Tracks are considered factors of a Release. Fields with categorical values (Format, Country & Styles) are formatted using One-Hot encoding, using only Releases from the given Wantlist and/or Collection.  Next, a PCA transformation is applied on these Releases, before applying the transformation on all extracted Releases from Discogs. Note that during this process, only the Styles within the Wantlist and/or Collection are kept in the database as Releases with other styles are most likely not interesting.

Artists, Labels, and Companies are considered to be groups of Releases, so to incorporate these, the mean and variance of the grouped PCA data is taken and attached to the original PCA data. In the current version, collaborating groups (e.g. two Artists together) are seen as a single entity, but this will be updated in future versions.

The Wantlist and/or Collection are seen as positive predictors, but negative predictors are usually not saved. Therefore, a random set of Releases of equal size as the positive predictors is taken as negative predictors. This introduces bias and thus, this package combines 10 models with 10 different negative predictors and multiplies the probabilities to obtain a single score for each Release. Note that Releases part of the Wantlist and/or Collection are not returned in the predictions.  










