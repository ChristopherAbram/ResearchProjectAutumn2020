import fiona
import os
import matplotlib.pyplot as plt
import shapely
import shapely.ops as so

from shapely.geometry import shape
from utils.definitions import get_project_path


def get_shapes(country_of_interest='Nigeria'):
    """
    Extracts shapes of admin 1 level for a given country.

    Returns regions, a dictionary where key=GDLcode(e.g. NGAr101)
    and value=shape geometry(e.g. shapely.geometry.multipolygon.MultiPolygon or Polygon)
    """
    shapefile_name = 'GDL Shapefiles V4.shp' # all the countries in the world
    shapefile_path = os.path.join(get_project_path(), 'data/shapefiles', shapefile_name)

    with fiona.open(shapefile_path) as shapes:

        regions = {}

        for shp in shapes:
            if(shp['properties']['country']== country_of_interest):
                regions[shp['properties']['GDLcode']] = shp['geometry']
    return regions

def plot_shapes(regions):
    """
    Plots all the shapes of a country (all the regions).
    Some regions have more than one shape(e.g. region which includes islands)
    and in that case we get MultiPolygon instead of just one Polygon.
    """
    for gdl_code, geo in regions.items():
        shapely_geometry = shape(geo)
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

    