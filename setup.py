# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, find_packages

setup(
    name="hy3dgen",
    version="2.1.0",
    url="https://github.com/Tencent/Hunyuan3D-2",
    packages=find_packages(),
    include_package_data=True,
    package_data={"hy3dgen": ["assets/*", "assets/**/*"]},
    install_requires=[
        'gradio',
        "tqdm>=4.66.3",
        'numpy',
        'ninja',
        'diffusers',
        'pybind11',
        'opencv-python',
        'einops',
        "transformers>=4.48.0",
        'omegaconf',
        'trimesh',
        'pymeshlab',
        'pygltflib',
        'xatlas',
        'accelerate',
        'gradio',
        'fastapi',
        'uvicorn',
        'rembg',
        'onnxruntime'
    ]
)
