from setuptools import find_packages, setup

package_name = "model_utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={"model_utils.pi05_export": ["quantization_profiles/*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "inference_manifest",
        "inference_service",
        "onnx",
        "onnxsim",
        "torch",
        "torchvision",
        "tqdm",
        "Pillow",
        "PyYAML",
        "safetensors>=0.4.3,<1.0.0",
    ],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="lwh",
    maintainer_email="liuweihong8@huawei.com",
    description="Model utilities for exporting and comparing ONNX models",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "frame_inspect = model_utils.frame_inspect:main",
            "observation-batch = model_utils.observation_batch_cli:main",
            "bump-inference-bundle-revision = model_utils.bundle_revision:main",
            "package-compiled-deployment = model_utils.package_compiled_deployment:main",
            "package-hmm-deployment = model_utils.hmm_export:main",
            "package-torch-deployment = model_utils.package_torch_deployment:main",
            "pi05-export = model_utils.pi05_export.__main__:console_main",
            "pi05-curvature-schedule = model_utils.pi05_export.curvature_schedule:main",
            "pi05-om-dump = model_utils.pi05_om_dump:main",
            "pi05-tune-schedule = model_utils.pi05_export.tune_schedule:main",
            "graspgen-export-onnx = model_utils.graspgen_export.export_onnx:main",
            "graspgen-onnx-to-om = model_utils.graspgen_export.onnx2om:main",
        ],
    },
)
