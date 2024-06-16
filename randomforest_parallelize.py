import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt
import seaborn as sns

artists = pd.read_csv("artists.csv")

artists = artists. \
    dropna(). \
    reset_index(drop=True)

artists['num_genres'] = artists.genres.map(len)

features = ["followers", "num_genres"]

start = time.time()
forest_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_split=40,
    max_depth=3,
    verbose=1,
    n_jobs=-1
)

forest_model.fit(artists[features],
                 artists.popularity)
end = time.time()
print(end - start)
