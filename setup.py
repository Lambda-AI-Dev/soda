import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soda",
    version="0.0.1",
    author="Tianyi Miao",
    author_email="mtianyi@seas.upenn.edu",
    description="A collection of state-of-the-art machine learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miaotianyi/soda",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
