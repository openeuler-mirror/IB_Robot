 #!/bin/bash

deploy_path=$1
arch=$(uname -m)

if [ ! -d "$deploy_path" ]; then
    echo "deploy_path does not exist"
    exit 1
fi


pip install funasr humanfriendly

cd ${deploy_path}

if [ "$arch" == "aarch64" ]; then
    onnxruntime_url=https://github.com/microsoft/onnxruntime/releases/download/v1.14.0/onnxruntime-linux-aarch64-1.14.0.tgz
    onnxruntime_name=onnxruntime-linux-aarch64-1.14.0
    ffmpeg_url=https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linuxarm64-gpl-shared.tar.xz
    ffmpeg_name=ffmpeg-master-latest-linuxarm64-gpl-shared
elif [ "$arch" == "x86_64" ]; then
    onnxruntime_url=https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/onnxruntime-linux-x64-1.14.0.tgz
    onnxruntime_name=onnxruntime-linux-x64-1.14.0
    ffmpeg_url=https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/ffmpeg-master-latest-linux64-gpl-shared.tar.xz
    ffmpeg_name=ffmpeg-master-latest-linux64-gpl-shared
else
    echo "unknown arch: ${arch}"
    exit 1
fi

if [ ! -d "$deploy_path/onnxruntime" ]; then
    wget ${onnxruntime_url} || exit 1
    tar -zxvf ${onnxruntime_name}.tgz || exit 1
    mv ${onnxruntime_name} onnxruntime
fi
if [ ! -d "$deploy_path/ffmpeg" ]; then
    wget ${ffmpeg_url} || exit 1
    tar -xvf ${ffmpeg_name}.tar.xz || exit 1
    mv ${ffmpeg_name} ffmpeg
fi

if [ ! -d "$deploy_path/FunASR" ]; then
    git clone https://github.com/alibaba-damo-academy/FunASR.git
fi

if [ ! -d "$deploy_path/FunASR/runtime/websocket/build/bin" ]; then
    cd $deploy_path/FunASR/runtime/websocket
    mkdir build
    cd build
    cmake  -DCMAKE_BUILD_TYPE=release .. -DONNXRUNTIME_DIR=${deploy_path}/onnxruntime -DFFMPEG_DIR=${deploy_path}/ffmpeg -DCMAKE_LINK_SEARCH_START_STATIC=OFF -DCMAKE_LINK_SEARCH_END_STATIC=OFF || exit 1
    make -j 4 || exit 1
fi 
