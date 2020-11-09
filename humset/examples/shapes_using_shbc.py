import humset.shapefiles_by_country as sbc

regions = sbc.get_shapes('Denmark')
sbc.plot_shapes(regions)