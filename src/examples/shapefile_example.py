import fiona
import os
import matplotlib.pyplot as plt
import shapely
import shapely.ops as so

from shapely.geometry import shape
from utils.definitions import get_project_path

shapefile_name = 'GDL Shapefiles V4.shp' # all the countries in the world
country_of_interest = 'Nigeria'
shapefile_path = os.path.join(get_project_path(), 'data/shapefiles', shapefile_name)

with fiona.open(shapefile_path) as shapes:

    regions = {}

    for shp in shapes:
        if(shp['properties']['country']== country_of_interest):
            regions[shp['properties']['GDLcode']] = shp['geometry']

    for gdl_code, geo in regions.items():
         
         
        print(gdl_code)
        shapely_geometry = shape(geo)
        print(type(shapely_geometry))
        if(isinstance(shapely_geometry, shapely.geometry.multipolygon.MultiPolygon)):
            fig, axs = plt.subplots()
            fig.suptitle(gdl_code)
            axs.set_aspect('equal', 'datalim')
            for geom in shapely_geometry.geoms:
                x, y = geom.exterior.xy
                axs.fill(x, y)
            plt.show()
        else:
            x, y = shapely_geometry.exterior.xy
            fig, ax = plt.subplots()
            fig.suptitle(gdl_code)
            ax.fill(x, y)
            plt.show()

    