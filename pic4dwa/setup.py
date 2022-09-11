from setuptools import setup

package_name = 'pic4dwa'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='eiraleandrea@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pic4dwa_ros = pic4dwa.pic4dwa_ros:main',
            'pic4dwa_omni_ros = pic4dwa.pic4dwa_omni_ros:main',
        ],
    },
)
