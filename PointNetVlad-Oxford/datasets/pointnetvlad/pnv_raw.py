import numpy as np
import os
import json

from datasets.base_datasets import PointCloudLoader


class PNVPointCloudLoader(PointCloudLoader):
    def set_properties(self):
        # Point clouds are already preprocessed with a ground plane removed
        self.remove_zero_points = False
        self.remove_ground_plane = False
        self.ground_plane_level = None

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        # Returns Nx3 ndarray
        file_path = os.path.join(file_pathname)
        pc = np.fromfile(file_path, dtype=np.float64)
        pc = np.float32(pc)
        # coords are within -1..1 range in each dimension
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        ############################  添加欧式聚类的结果 12*3  ##################
        graph_path = file_path.replace('benchmark_datasets', 'graphs')
        graph_path = graph_path.replace('bin', 'json')
        graph_data = json.load(open(graph_path))
        centers_data = np.array(graph_data["centers"])
        pc = np.concatenate((pc, centers_data), axis=0)
        return pc  #  4096 + 12 = 4108
