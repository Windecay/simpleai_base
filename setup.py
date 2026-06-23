from setuptools import setup

setup(
    name='simpleai_base',
    version='0.3.47',
    packages=['simpleai_base'],
    install_requires=[
        'cbor2',
    ],
    python_requires='>=3.13',
    include_package_data=True,
    description='A Python package with Rust code',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/my_python_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.13',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
