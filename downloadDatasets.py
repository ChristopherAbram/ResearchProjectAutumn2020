import requests
import pathlib

urls = {
    'population_nga_2018-10-01': 'https://data.humdata.org/dataset/62ec6c48-2f23-476b-8c1e-e924ad79908d/resource/ec1ac1b2-616e-43a7-ba8c-29eaf3479f24/download/population_nga_2018-10-01.zip',
    'population_zaf_2019-07-01_geotiff': 'https://data.humdata.org/dataset/cbfc4206-35c8-42d4-a096-b2dd0aec983d/resource/d0ff7101-ba3e-490a-8fe9-57b9b6a0aa1f/download/population_zaf_2019-07-01_geotiff.zip'
}

storageFolderPath = './datasets'

pathlib.Path(storageFolderPath).mkdir(exist_ok=True)

for datasetName, url in urls.items():
    r = requests.get(url)

    with open(f'{storageFolderPath}/{datasetName}', 'wb') as f:
        f.write(r.content)
