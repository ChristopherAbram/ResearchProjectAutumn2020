import os
import zipfile
import requests
import shutil
import urllib.request as request
from contextlib import closing

datasets = {
    'humdata':
        ['https://data.humdata.org/dataset/62ec6c48-2f23-476b-8c1e-e924ad79908d/resource/ec1ac1b2-616e-43a7-ba8c-29eaf3479f24/download/population_nga_2018-10-01.zip'],
    'worldpop':
        ['ftp://ftp.worldpop.org.uk/GIS/Population/Global_2000_2020/2015/NGA/nga_ppp_2015.tif'],
    'grid3':
        ['https://s3-eu-west-1.amazonaws.com/files.grid3.gov.ng/pop/GRID3+-+NGA+-+National+Population+Data+-+v1.2.zip'],
    'shapefiles':
        ['https://globaldatalab.org/assets/2020/03/GDL%20Shapefiles%20V4.zip']
    }

script_path = os.path.dirname(os.path.abspath(__file__))

for dirname, urls in datasets.items():
    current_path = os.path.join(script_path, 'data', dirname)
    os.makedirs(current_path, exist_ok=True)
    for url in urls:
        filename = url.split('/')[-1]

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