import os
import numpy as np
import pandas as pd

from humset.utils.definitions import get_project_path


def get_shdi(country_iso_code='NGA'):
    hdi_path = os.path.join(get_project_path(), 'data', 'shdi', 'SHDI Complete 4.0 (1).csv')
    data = pd.read_csv(hdi_path, usecols=['iso_code', 'year', 'level', 'GDLCODE', 'shdi'])
    data = data.set_index('iso_code', drop=False)
    data = data.loc[
        (data['iso_code'] == country_iso_code) & 
        (data['year'] == 2018) & 
        (data['level'] == 'Subnat')]
    return np.array(data.loc[:,['GDLCODE', 'shdi']])