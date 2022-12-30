
# Netflix Chatbot: Extracting, Transforming, & Loading Data

## Stephanie Fissel

# Import Packages


```python
# import packages
import pandas as pd
import pymongo
import requests
import json
```

# PART 1: Extract and Transform the Data

# Extract data source into one Pandas dataframe:
## Import relevant csv files as dataframes


```python
best_movies = pd.read_csv("/Users/stephaniefissel/Desktop/ds2002/final project data/Best Movies Netflix.csv")
best_shows = pd.read_csv("/Users/stephaniefissel/Desktop/ds2002/final project data/Best Shows Netflix.csv")
```

## Add dummy variables for movies and shows to distinguish type


```python
best_movies['MOVIE_DUMMY'] = 1
best_movies['SHOW_DUMMY'] = 0
best_movies['NUMBER_OF_SEASONS'] = 0
best_shows['MOVIE_DUMMY'] = 0
best_shows['SHOW_DUMMY'] = 1
```

## Concatenate dataframes with best movies and shows into Pandas dataframe


```python
best = [best_movies, best_shows]
all = pd.concat(best, axis = 0, sort = False)
all = all.drop('index', axis=1)
```

## Convert column variable integers to strings


```python
all[['RELEASE_YEAR', 'SCORE', 'NUMBER_OF_VOTES', 'DURATION', 'MOVIE_DUMMY', 'SHOW_DUMMY', 'NUMBER_OF_SEASONS']] = all[['RELEASE_YEAR', 'SCORE', 'NUMBER_OF_VOTES', 'DURATION', 'MOVIE_DUMMY', 'SHOW_DUMMY', 'NUMBER_OF_SEASONS']].astype(str)
all.index = all.index.map(str)
```

# PART 2: Load Data into MongoDB

## Connect to MongoDB Instance


```python
host_name = "localhost"
port = "27017"

atlas_cluster_name = "sandbox"
atlas_default_dbname = "local"
```


```python
conn_str = {
    "local" : f"mongodb://{host_name}:{port}/",
}

client = pymongo.MongoClient(conn_str["local"])

print(f"Local Connection String: {conn_str['local']}")
```

    Local Connection String: mongodb://localhost:27017/


## Assign database


```python
db = client['Netflix']
```

## Store in Mongo DB


```python
db.collection.insert_many(all.to_dict('records'))
```




    <pymongo.results.InsertManyResult at 0x7fa0a6de5988>


