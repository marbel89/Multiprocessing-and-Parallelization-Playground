import time
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

"""
This script demonstrates how to load and preprocess a dataset,
train a RandomForestRegressor model, and use a cached function to get model predictions efficiently. 
The @lru_cache decorator helps speed up repeated predictions by caching the results of previous computations
"""

artists = pd.read_csv("artists.csv")

artists = artists.dropna().reset_index(drop=True)

artists['num_genres'] = artists.genres.map(len)
features = ["followers", "num_genres"]

start = time.time()
forest_model = RandomForestRegressor(n_estimators=300,
                                     min_samples_split=40,
                                     max_depth=3,
                                     verbose=1,
                                     n_jobs=-1
                                     )

forest_model.fit(artists[features], artists.popularity)
end = time.time()
print("Model training runtime: ", end - start)


@lru_cache
def get_model_predictions(model, _inputs):
    """
       Get model predictions for given inputs using a cached function.

       Parameters
       ----------
       model : RandomForestRegressor
           The trained machine learning model.
       _inputs : tuple
           A tuple containing the input values for 'followers' and 'num_genres'.

       Returns
       -------
       numpy.ndarray
           The predicted values from the model.
       """
    formatted_inputs = pd.DataFrame.from_dict({
        "followers": inputs[0],
        "num_genres": inputs[1]},
        orient="index").transpose()

    return model.predict(formatted_inputs)


inputs = (2, 4)

start = time.time()
get_model_predictions(forest_model, inputs)
end = time.time()
print("Model predictions runtime (w/o) cache: ", end - start)

start = time.time()
get_model_predictions(forest_model, inputs)
end = time.time()
print("Model predictions runtime (w/) cache: ", end - start)
