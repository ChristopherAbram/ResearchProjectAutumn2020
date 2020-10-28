import os
import sys

from utils.definitions import get_project_path
from visualization import AlignMapsEditor


def main(argc, argv):

    country = 'NGA'
    in_files = [
        os.path.join(get_project_path(), "data", "humdata",
                     'population_%s_2018-10-01.tif' % country.lower()),
        os.path.join(get_project_path(), "data", "worldpop",
                     '%s_ppp_2015.tif' % country.lower()),
        os.path.join(get_project_path(), "data", "grid3",
                     '%s - population - v1.2 - mastergrid.tif' % country)
    ]

    # lat, lon = (6.541456, 3.312719)  # Lagos
    lat, lon = (8.499714, 3.423570) # Ago-Are
    # lat, lon = (7.382932, 3.929635) # Ibadan
    # lat, lon = (4.850891, 6.993961) # Port Harcourt

    editor = AlignMapsEditor(in_files[0], in_files[2], 'HUMDATA', 'GRID3', (lat, lon))
    editor.wait()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
