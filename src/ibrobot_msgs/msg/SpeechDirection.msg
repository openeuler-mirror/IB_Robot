# 人声方向消息(REP-103 平面角度)
#
# 发布契约(方向输出事件,无人声不发布):
#   - 人声开始后获得稳定初始估计时发布 voice_begin 方向,降低交互响应延迟
#   - 长语音累计达到上限时发布中间方向,降低持续讲话场景的响应延迟
#   - 一段人声结束时发布段末方向；三类方向均为有效控制输入
#   - 无人声时不发布任何方向消息
#   - 不周期性空转发布

std_msgs/Header header      # stamp=本次估计使用的最后一个音频样本时间;frame_id=base_link
float32 azimuth_rad         # REP-103 平面角度:0=前,+π/2=左,-π/2=右,左为正
uint32 seq_id               # 方向输出事件序号,每次早期/中间/段末输出递增;消费者用于去重。uint32 翻转后从 0 重新开始
uint32 segment_id           # 同一段人声的稳定标识;voice_begin/seg_end 使用同一值
string direction_type       # voice_begin / mid_long_seg / seg_end
