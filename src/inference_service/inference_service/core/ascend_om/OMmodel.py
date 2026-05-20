"""
OMmodel.py

om 模型的封装调用
该程序是基于官方文档并做了一点点修改编写的通用 om 加载与推理程序
与不同 om 模型的不同输入输出格式无关
"""

try:
    import acl
except ImportError:  # pragma: no cover - only available on Ascend runtime hosts
    acl = None
import contextlib

import numpy as np

ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def logger(msg: str):
    print(f"[OM_model]: {msg}")


class OMmodel:
    def __init__(self, model_path):
        if acl is None:
            raise RuntimeError("Ascend ACL runtime is required for OM inference")
        self._closed = False
        logger(f"model path: {model_path}")
        self.device_id = 0

        ret = acl.init()
        self.check_ret(ret, "Failed to init")

        ret = acl.rt.set_device(self.device_id)
        self.check_ret(ret, "Failed to create device")
        logger(f"set device id {self.device_id}, ret {ret}")

        self.model_id, ret = acl.mdl.load_from_file(model_path)
        self.check_ret(ret, "Failed to load model from file")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        self.check_ret(ret, "Failed to get desc")

        self.input_dataset, self.input_data = self.prepare_dataset("input")
        self.output_dataset, self.output_data = self.prepare_dataset("output")

    def forward(self, inputs):
        # TEMP: ensure the current thread holds the ACL device context
        ret = acl.rt.set_device(self.device_id)
        self.check_ret(ret, "Failed to set device in forward()")
        input_num = len(inputs)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            ret = acl.rt.memcpy(
                self.input_data[i]["buffer"],
                self.input_data[i]["size"],
                bytes_ptr,
                len(bytes_data),
                ACL_MEMCPY_HOST_TO_DEVICE,
            )
            self.check_ret(ret, "Failed to memcpy from host to device")

        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        self.check_ret(ret, "Failed to execute forward")

        inference_result = []
        for i, _ in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])

            ret = acl.rt.memcpy(
                buffer_host,
                self.output_data[i]["size"],
                self.output_data[i]["buffer"],
                self.output_data[i]["size"],
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            self.check_ret(ret, "Failed to memcpy from device to host")
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])

            data = np.frombuffer(bytes_out, dtype=np.float32).copy()
            inference_result.append(data)

            ret = acl.rt.free_host(buffer_host)
            self.check_ret(ret, "Failed to free host")

        return inference_result

    def close(self):
        if self._closed or acl is None or not hasattr(self, "input_data"):
            return
        self._closed = True
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                acl.destroy_data_buffer(item["data"])
                acl.rt.free(item["buffer"])
        acl.mdl.destroy_dataset(self.input_dataset)
        acl.mdl.destroy_dataset(self.output_dataset)
        acl.mdl.destroy_desc(self.model_desc)
        acl.mdl.unload(self.model_id)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def prepare_dataset(self, io_type):
        if io_type == "input":
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index

        dataset = acl.mdl.create_dataset()
        data = []
        for i in range(io_num):
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            self.check_ret(ret, "Prepare dataset: Failed to malloc")

            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            self.check_ret(ret, "Prepare dataset: Failed to add dataset buffer")
            data.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, data

    def check_ret(self, ret, msg):
        if ret != 0:
            raise Exception(f"{msg}, Error code: {ret}")
