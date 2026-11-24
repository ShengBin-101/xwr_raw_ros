import os
from glob import glob

from setuptools import setup, find_packages

package_name = 'radar_pipelines'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'test.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Per-chirp-profile calibration sidecars, which core/calib.py loads
        # by profile name at run time.
        (os.path.join('share', package_name, 'calib'), glob('config/calib/*.yaml')),
    ],
    # numpy, scipy, numba, opencv and yaml are not listed here. They are
    # declared as apt exec_depends in package.xml (see the comment there).
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chan Sheng Bin',
    maintainer_email='ShengBin-101@users.noreply.github.com',
    description='Processing nodes for the TI IWR6843AOP raw-data driver: MMVR-layout '
                'dual-view heatmaps, a CFAR point cloud and heatmap viewers.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'heatmap_node.py = radar_pipelines.nodes.heatmap_node:main',
            'pointcloud_node.py = radar_pipelines.nodes.pointcloud_node:main',
            'capture_background.py = radar_pipelines.nodes.capture_background:main',
            'elevation_calibration.py = radar_pipelines.nodes.elevation_calibration:main',
            'view_heatmaps.py = radar_pipelines.nodes.view_heatmaps:main',
            'heatmap_viz.py = radar_pipelines.nodes.heatmap_viz:main',
        ],
    },
)
