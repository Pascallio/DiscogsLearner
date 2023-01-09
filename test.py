# import discogslearner

# if __name__ == "__main__":
#     output_file = r"C:\Users\Pascal Maas\Documents\GitHub\DiscogsLearner\Data/discogs_db.tsv"
#     my_genre = "Electronic"
#     my_token = "your_token_here"
    
#     extracter = discogslearner.Extracter(genre = my_genre, url = "https://discogs-data-dumps.s3-us-west-2.amazonaws.com/data/2022/discogs_20221201_releases.xml.gz")
#     extracter.extract(output = output_file)
#     learner = discogslearner.Learner(db_path = output_file, 
#                                     use_wantlist=True, 
#                                     use_collection=True,
#                                     token = my_token)

#     outcome = learner.learn_and_predict(n_models = 10)
#     print(outcome)

import discogslearner
extracter = discogslearner.Extracter()
extracter.createOneHotFiles("Data")