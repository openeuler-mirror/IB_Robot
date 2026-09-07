import os
from glob import glob

from setuptools import find_packages, setup

package_name = "voice_asr_service"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # launch 入口统一为 *.launch.py;被 launch 文件 import 的辅助模块
        # 应放在包目录(voice_asr_service/)而非 launch/,避免被此处 data_files 误捕获
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="IB-Robot Team",
    maintainer_email="dev@example.com",
    description="Voice ASR Service for IB-Robot using sherpa-onnx",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "voice_asr_node = voice_asr_service.voice_asr_node:main",
            "package_sherpa_asr_bundle = voice_asr_service.package_sherpa_asr_bundle:main",
            "package_silero_vad_bundle = voice_asr_service.package_silero_vad_bundle:main",
            "package_fullsubnet_bundle = voice_asr_service.package_fullsubnet_bundle:main",
            "speech_direction_node = voice_asr_service.speech_direction.node:main",
            "speech_direction_report = voice_asr_service.speech_direction.report_cli:main",
        ],
    },
)
