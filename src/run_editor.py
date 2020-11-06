import os, sys

from utils.definitions import get_dataset_paths
from visualization import AlignMapsEditor


def main(argc, argv):
    in_files = get_dataset_paths('NGA')

    # lat, lon = (6.541456, 3.312719)  # Lagos
    lat, lon = (8.499714, 3.423570) # Ago-Are
    # lat, lon = (7.382932, 3.929635) # Ibadan
    # lat, lon = (4.850891, 6.993961) # Port Harcourt

    editor = AlignMapsEditor(in_files['humdata'], in_files['grid3'], (lat, lon))
    editor.wait()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))