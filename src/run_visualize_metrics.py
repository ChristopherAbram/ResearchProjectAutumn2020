import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils.definitions import get_dataset_paths, get_project_path
from utils.raster import RasterTable
from visualization import AlignMapsEditor


def main(argc, argv):
    in_files = get_dataset_paths('NGA')
    names = ['True positive', 'False positive', 'False negative', 
        'True negative', 'Accuracy', 'Recall', 'Precision', 'F1 Score']
    
    run_editor = True
    
    m_inx = 2
    metrics_path = [
        os.path.join(get_project_path(), 'data', 'results', 'nga_metrics_p30_t20.tif'),
        os.path.join(get_project_path(), 'data', 'results', 'nga_metrics_p30_t50.tif'),
        os.path.join(get_project_path(), 'data', 'results', 'nga_metrics_p30_t80.tif')
    ]
    
    if not run_editor:
        # Plot the entire dataset:
        fig2, ax2 = plt.subplots(2, 2, figsize=(15, 10))
        fig1, ax1 = plt.subplots(2, 2, figsize=(15, 10))

        table = RasterTable(metrics_path[m_inx], 1, 1)
        for layer, ax in enumerate(ax1.ravel()):
            ax.imshow(table(layer + 1).get(0, 0).data, cmap='magma', norm=LogNorm())
            ax.set_title(names[layer])

        for layer, ax in enumerate(ax2.ravel()):
            ax.imshow(table(4 + layer + 1).get(0, 0).data, cmap='jet')
            ax.set_title(names[4 + layer])
        plt.show()

    else:
        # lat, lon, zoom = (6.541456, 3.312719, 109) # Lagos
        # lat, lon, zoom = (6.457581, 3.380313, 54) # Lagos
        lat, lon, zoom = (8.499714, 3.423570, 27) # Ago-Are
        # lat, lon, zoom = (7.382932, 3.929635, 432) # Ibadan
        # lat, lon, zoom = (4.850891, 6.993961, 109) # Port Harcourt
        # lat, lon, zoom = (11.961825, 8.540164, 213) # Kano

        editor = AlignMapsEditor(
            in_files['humdata'], metrics_path[m_inx], (lat, lon, zoom), 
            'HUMDATA', 'NORM', index2=5) # 5 is accuracy
        editor.wait()

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
