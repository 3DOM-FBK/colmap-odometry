from glob import glob
import os

from setuptools import find_packages, setup


package_name = "colmap_odometry"

setup(
    name=package_name,
    version="1.0.0",
    py_modules=["main"],
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "calibration"), glob("calibration/*.yaml")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "rosdep"), glob("rosdep/*.yaml")),
    ],
    package_data={
        package_name: [
            "thirdparty/ALIKED/models/*.pth",
            "thirdparty/SuperPoint_open/weights/*.pth",
        ],
    },
    scripts=[
        "scripts/colmap_odometry_node",
        "scripts/colmap_odometry_offline",
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="ROS 2-enabled Colmap-Odometry visual odometry package.",
    license="BSD-3-Clause",
)
