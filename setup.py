from setuptools import setup, find_packages

setup(
    name="hivg",
    version="0.1.0",
    description="HiVG: Hierarchical SVG Tokenization for Scalable Vector Graphics Generation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ximinng",
    url="https://github.com/ximinng/HiVG",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    package_data={
        "hivg_tokenizer": ["configs/*.yaml"],
    },
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pyyaml>=5.0",
        "pillow",
        "cairosvg",
        "fire",
        "tqdm",
        "numpy>=1.26",
        "piqa",
    ],
    extras_require={
        "vllm": ["vllm>=0.4.0"],
    },
)
