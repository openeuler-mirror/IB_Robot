"""speech_direction 公共数据类型。

speech_direction 链路需要的数据结构。
"""

from __future__ import annotations

from typing import TypedDict


class SpeechDirectionResult(TypedDict):
    """get_speech_direction 返回的段级方向结果。

    以 TypedDict 形式声明 runtime.get_speech_direction() 实际返回的 dict 契约，
    字段与 runtime 构造的 dict 一一对应，避免 dataclass 与 dict 字段漂移
    (历史教训:dict 曾新增 type 键而 dataclass 未同步)。

    Attributes:
        wall_clock_ts: 方向产生时刻(墙钟秒,time.time 域,用于 age_ms 计算;
            与 ROS 时钟 get_clock().now 不同域,见 node.py stamp 构造注释)
        angle: 人声方向角(度,阵列坐标系:0°=右,90°=前,180°=左,逆时针为正);
               None 表示无可靠方向(冷启动/过期/无人声)
        is_speech: 当前是否在说话(透传供调用方参考,不参与 angle 判定)
        age_ms: 方向距今时长(毫秒)
        seq_id: 段序号,每段递增,用于消费者去重
        type: 方向输出类型("mid_long_seg"=长语音中间方向 / "seg_end"=段末方向 / None=未知)
    """

    wall_clock_ts: float
    angle: float | None
    is_speech: bool
    age_ms: float
    seq_id: int
    type: str | None
    segment_id: int


__all__ = ["SpeechDirectionResult"]
