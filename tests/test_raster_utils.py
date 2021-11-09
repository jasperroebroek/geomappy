import os
import shutil


class TestRaster:
    def __enter__(self):
        try:
            os.mkdir("_test_rasters")
        except FileExistsError:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree("_test_rasters")