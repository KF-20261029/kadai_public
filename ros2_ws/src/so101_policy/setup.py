from setuptools import find_packages, setup

package_name = 'so101_policy'

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
    maintainer='SO-101 task contributors',
    maintainer_email='maintainers@example.invalid',
    description='ROS 2 interface for the trained SO-101 pi0.5 policy.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'check_model = so101_policy.model:main',
            'check_observation = so101_policy.observation:main',
            'check_prediction = so101_policy.prediction:main',
            'execute_chunk = so101_policy.control:main',
            'execute_realtime = so101_policy.realtime:main',
        ],
    },
)
