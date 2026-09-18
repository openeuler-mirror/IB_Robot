import numpy as np

from hardware_mock.contract_mock_node import _array_to_image


def test_array_to_image_packs_contiguous_bgr_data() -> None:
    frame = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    message = _array_to_image(frame[:, ::-1], encoding="bgr8")

    expected = np.ascontiguousarray(frame[:, ::-1])
    assert (message.height, message.width, message.step) == (3, 4, 12)
    assert message.encoding == "bgr8"
    assert bytes(message.data) == expected.tobytes()
