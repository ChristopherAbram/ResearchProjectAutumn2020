import fiona
import os
import matplotlib.pyplot as plt
import matplotlib
import shapely
import shapely.ops as so

from shapely.geometry import shape
from humset.utils.definitions import get_project_path


def get_shapes(country_of_interest='Nigeria'):
    """
    Extracts shapes of admin 1 level for a given country.
    Shapes source SHDI shape files(maps)
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

def get_shapes_rasterstats(country_of_interest='Nigeria'):
    """
    Extracts shapes of admin 1 level for a given country.
    Shapes source SHDI shape files(maps)
    Returns regions, a list of original shapes grouped by a country
    """
    shapefile_name = 'GDL Shapefiles V4.shp' # all the countries in the world
    shapefile_path = os.path.join(get_project_path(), 'data/shapefiles', shapefile_name)

    with fiona.open(shapefile_path) as shapes:

        regions = []

        for shp in shapes:
            if(shp['properties']['country']== country_of_interest):
                regions.append(shp)
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
    plt.show()

def plot_whole_country(regions, metric_title, metric_per_region):
    """
    Plots all the shapes of a country (all the regions).
    Some regions have more than one shape(e.g. region which includes islands)
    and in that case we get MultiPolygon instead of just one Polygon.
    """
    cmap = matplotlib.cm.get_cmap('viridis_r')
    fig, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    for gdl_code, geo in regions.items():
        shapely_geometry = shape(geo)
        if(isinstance(shapely_geometry, shapely.geometry.multipolygon.MultiPolygon)):
            ax.set_aspect('equal', 'datalim')
            for geom in shapely_geometry.geoms:
                x, y = geom.exterior.xy
                hdi = metric_per_region[gdl_code]
                color = cmap(hdi)
                ax.fill(x, y, facecolor=color)
        else:
            x, y = shapely_geometry.exterior.xy
            hdi = metric_per_region[gdl_code]
            color = cmap(hdi)
            ax.fill(x, y, facecolor=color)
    fig.suptitle(metric_title)
    norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1)
    ax2 = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm)
    plt.show()