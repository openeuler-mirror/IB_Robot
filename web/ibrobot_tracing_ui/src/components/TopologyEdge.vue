<script setup lang="ts">
import { computed } from "vue";
import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, type EdgeProps } from "@vue-flow/core";

interface TopologyEdgeData {
  displayLabel: string;
  displayMetric: string;
  directed: boolean;
}

const props = defineProps<EdgeProps<TopologyEdgeData>>();

const path = computed(() =>
  getSmoothStepPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    sourcePosition: props.sourcePosition,
    targetX: props.targetX,
    targetY: props.targetY,
    targetPosition: props.targetPosition,
    borderRadius: 8,
    offset: 28,
  }),
);

const labelPosition = computed(() => ({
  transform: `translate(-50%, -100%) translate(${path.value[1]}px, ${path.value[2] - 8}px)`,
}));
</script>

<template>
  <BaseEdge
    :id="id"
    :path="path[0]"
    :marker-start="data.directed ? markerStart : undefined"
    :marker-end="data.directed ? markerEnd : undefined"
    :interaction-width="24"
  />
  <EdgeLabelRenderer v-if="data.displayLabel">
    <div class="topology-edge-label" :style="labelPosition" :title="data.displayLabel">
      <span>{{ data.displayLabel }}</span>
      <small v-if="data.displayMetric">{{ data.displayMetric }}</small>
    </div>
  </EdgeLabelRenderer>
</template>
