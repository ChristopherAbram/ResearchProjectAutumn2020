import sys, os

from humset.utils.definitions import get_project_path, get_dataset_paths
from humset.metrics import RasterTableScheduler, SimpleMetrics, AggregatedMetrics


def main(argc, argv):
    country = 'NGA'
    # in_files = get_dataset_paths(country)
    in_files = {
        'humdata': os.path.join(get_project_path(), 'data', 'out', 'example_humdata.tif'),
        'grid3': os.path.join(get_project_path(), 'data', 'out', 'example_grid3.tif')
    }

    patch_size = 60
    thresholds = [0.2, 0.5, 0.8]
    threads = 12
    impl = SimpleMetrics

    for threshold in thresholds:
        print("Compute metrics for patch_size=", patch_size, ", threshold=", threshold)
        # Build output filename:
        dirpath = os.path.join(get_project_path(), "data", "out")
        filename = '%s_metrics_p%d_t%d.tif' % (country.lower(), patch_size, int(threshold * 100))

        scheduler = RasterTableScheduler(
            in_files['humdata'], in_files['grid3'], 
            patch_size, threshold, threads, fake=False, metrics_impl=impl)

        scheduler.run()
        scheduler.save(dirpath, filename)

    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
