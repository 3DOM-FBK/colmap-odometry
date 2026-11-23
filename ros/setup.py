from glob import glob
import os

from setuptools import find_packages, setup

package_name = "colmap_odometry"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "rosdep"), glob("rosdep/*.yaml")),
    ],
    # Mandatory: numpy, opencv (>= 4.5, USAC_MAGSAC), pycolmap (>= 3.12), pyyaml.
    install_requires=["setuptools", "numpy", "pycolmap>=3.12", "pyyaml"],
    # Feature libraries (LightGlue with ALIKED/SuperPoint/DISK, LoMa) are git
    # submodules in colmap_odometry/thirdparty; these are their dependencies.
    extras_require={
        "features": ["torch", "torchvision", "kornia", "einops", "pillow"],
        # Offline runs and regression checks read bags without ROS.
        "offline": ["rosbags"],
        "test": ["pytest"],
    },
    # Ship the vendored sources for non-symlink installs.
    package_data={
        package_name: [
            "thirdparty/LightGlue/lightglue/*.py",
            "thirdparty/LoMa/src/loma/**/*.py",
        ],
    },
    entry_points={
        "console_scripts": [
            "colmap_odometry_node = colmap_odometry.ros.node:main",
            "colmap_odometry_offline = colmap_odometry.offline:main",
            "colmap_odometry_check = colmap_odometry.check:main",
        ],
    },
    zip_safe=True,
    maintainer="Pawel Trybala",
    maintainer_email="ptrybala@fbk.eu",
    description="Multi-camera visual odometry on COLMAP geometry with pluggable features.",
    license="BSD-3-Clause",
    tests_require=["pytest"],
)
