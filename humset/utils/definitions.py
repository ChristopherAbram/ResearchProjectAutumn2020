import os, json
from pathlib import Path

def get_project_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    # return Path(__file__).parent.parent 

def get_datasets_uri():
    with open(os.path.join(get_project_path(), 'datasets.json')) as datasets:
        return json.load(datasets)

def get_dataset_paths(country=None):
    cd = {
        'NGA': {
            'humdata': os.path.join(get_project_path(), "data", "humdata", 'population_nga_2018-10-01.tif'),
            'grid3': os.path.join(get_project_path(), "data", "grid3", 'NGA - population - v1.2 - mastergrid.tif'),
            'worldpop': os.path.join(get_project_path(), "data", "worldpop", 'nga_ppp_2015.tif'),
        },
    }
    return cd if country is None else cd[country]

def get_memory_limit():
    return 1 * 1024 * 1024 * 1024 # in bytes (0.5 GB)