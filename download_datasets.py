import os
import zipfile
import requests
import shutil
import urllib.request as request
from contextlib import closing
from urllib.parse import unquote
from humset.utils.definitions import get_datasets_uri

datasets = get_datasets_uri()
script_path = os.path.dirname(os.path.abspath(__file__))

for dirname, urls in datasets.items():
    current_path = os.path.join(script_path, 'data', dirname)
    os.makedirs(current_path, exist_ok=True)
    for url in urls:
        filename = unquote(url.split('/')[-1])
        # if(url.split('/')[-1] == '?levels=1%2B4&interpolation=0&extrapolation=0&nearest_real=0&format=csv'):
        #     filename = 'GDL-Sub-national-HDI-data.csv'
        # else:
        #     filename = url.split('/')[-1]

        # if file already exists, do nothing
        if os.path.exists(os.path.join(current_path, filename)):
            continue
        print(f'Downloading \t{filename}')

        # for ftp, use urllib
        if "ftp" in url:
            with closing(request.urlopen(url)) as r:
                with open(os.path.join(current_path, filename), 'wb') as f:
                    shutil.copyfileobj(r, f)

        # otherwise use requests
        else:
            r = requests.get(url)
            with open(os.path.join(current_path, filename), 'wb') as f:
                f.write(r.content)

        # if file is zipped, unzip
        if url.endswith('.zip'):
            with zipfile.ZipFile(os.path.join(current_path, filename), 'r') as zip_ref:
                zip_ref.extractall(current_path)
            
            os.remove(os.path.join(current_path, filename))
