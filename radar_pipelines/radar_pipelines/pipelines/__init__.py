"""The radar processing pipelines. One package, several pipelines.

  heatmap_rexo     MMVR-format dual-view Cartesian heatmaps (REXO model input)
  pointcloud_cfar  3D point cloud via selectable CFAR on a Capon range-azimuth map

Both are plain Python without rclpy, so they can be driven from a bag
replay, a notebook or a unit test as easily as from a node.
"""

from .heatmap_rexo import RexoHeatmapConfig, RexoHeatmapPipeline, RexoHeatmapResult
from .pointcloud_cfar import PointCloudConfig, PointCloudPipeline, PointCloudResult

__all__ = [
    'RexoHeatmapConfig', 'RexoHeatmapPipeline', 'RexoHeatmapResult',
    'PointCloudConfig', 'PointCloudPipeline', 'PointCloudResult',
]
