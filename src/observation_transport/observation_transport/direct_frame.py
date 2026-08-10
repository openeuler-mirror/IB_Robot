"""Compatibility facade for the public IB-Robot direct-frame ingress."""

from observation_transport.frame_ingress import (
    DirectFrameProducerError,
    DirectFrameStreamConfig,
    DirectFrameStreamDescriptor,
    DirectFrameStreamStatus,
    FrameAdmissionDisposition,
    FrameAdmissionReceipt,
    FrameIngress,
    FrameIngressError,
    FrameStreamSnapshot,
    FrameSubmissionReceipt,
    FrameTransportSnapshot,
    PreparedDirectFrame,
    QueuePolicy,
    StreamSessionView,
    create_frame_ingress,
)
from observation_transport.native_frame_ingress import NativeFrameIngress

DirectFrameProducer = NativeFrameIngress

__all__ = [
    "DirectFrameProducer",
    "DirectFrameProducerError",
    "DirectFrameStreamConfig",
    "DirectFrameStreamDescriptor",
    "DirectFrameStreamStatus",
    "FrameAdmissionDisposition",
    "FrameAdmissionReceipt",
    "FrameIngress",
    "FrameIngressError",
    "FrameStreamSnapshot",
    "FrameSubmissionReceipt",
    "FrameTransportSnapshot",
    "NativeFrameIngress",
    "PreparedDirectFrame",
    "QueuePolicy",
    "StreamSessionView",
    "create_frame_ingress",
]
