<script setup lang="ts">
import { Handle, Position } from "@vue-flow/core";
import { formatDuration } from "../utils/format";

export interface TopologyNodeData {
  label: string;
  componentId: string;
  kind: string;
  parentName: string;
  provenance: string;
  processingMs: number | null;
  scoped: boolean;
}

defineProps<{ data: TopologyNodeData; selected?: boolean }>();

const KIND_LABELS: Record<string, string> = {
  ros_node: "ROS 节点",
  module: "模块",
  operation: "操作",
  instant_event: "瞬时事件",
  data_source: "数据源",
  data_sink: "数据出口",
};
</script>

<template>
  <div class="topology-node" :class="[`kind-${data.kind}`, { scoped: data.scoped, selected }]">
    <Handle type="target" :position="Position.Left" />
    <div class="node-heading">
      <span>{{ KIND_LABELS[data.kind] || data.kind }}</span>
      <i v-if="data.provenance === 'observed'" title="由追踪数据动态发现">观测</i>
    </div>
    <strong :title="data.label">{{ data.label }}</strong>
    <code :title="data.componentId">{{ data.componentId }}</code>
    <div v-if="data.parentName || data.processingMs !== null" class="node-meta">
      <span v-if="data.parentName" :title="data.parentName">属于 {{ data.parentName }}</span>
      <b v-if="data.processingMs !== null">{{ formatDuration(data.processingMs) }}</b>
    </div>
    <Handle type="source" :position="Position.Right" />
  </div>
</template>
