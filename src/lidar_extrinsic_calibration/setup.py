from setuptools import find_packages, setup

package_name = 'lidar_extrinsic_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rm',
    maintainer_email='2200187107@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'integrate_pointcloud_to_pcd = lidar_extrinsic_calibration.integrate_pointcloud_to_pcd:main',
            'imu_calibration = lidar_extrinsic_calibration.imu_calibration:main',
            'P_gl_xy = lidar_extrinsic_calibration.P_gl_xy:main',
            'P_gl_x_y = lidar_extrinsic_calibration.P_gl_x_y:main',
            'estimate_delay_by_gyro_corr = lidar_extrinsic_calibration.estimate_delay_by_gyro_corr:main',
            'offline_extrinsic_calib_se2 = lidar_extrinsic_calibration.offline_extrinsic_calib_se2:main',
            'extrinsic_test = lidar_extrinsic_calibration.extrinsic_test:main',
            'timestamp_test = lidar_extrinsic_calibration.timestamp_test:main',
            
 
        ],
    },
)
