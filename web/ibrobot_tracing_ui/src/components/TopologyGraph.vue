<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { MarkerType, Panel, VueFlow, useVueFlow, type Edge, type Node, type NodeMouseEvent } from "@vue-flow/core";
import { Background } from "@vue-flow/background";
import { Controls } from "@vue-flow/controls";
import { MiniMap } from "@vue-flow/minimap";
import { useProfilerStore } from "../stores/profiler";
import { formatDuration, metricLabel, recordToData } from "../utils/format";
import AppIcon from "./AppIcon.vue";
import AsyncState from "./AsyncState.vue";
import TopologyEdge from "./TopologyEdge.vue";
import TopologyNode from "./TopologyNode.vue";

const store = useProfilerStore();
const { topology, analysisId, analysisState, analysisError, searchQuery, componentId, motionPaused, requestId, metric } = storeToRefs(store);
const nodes = ref<Node[]>([]);
const edges = ref<Edge[]>([]);
const layoutPending = ref(false);
const { fitView } = useVueFlow({ id: "ibrobot-topology" });
let elkPromise: ReturnType<typeof createElk> | null = null;
let layoutGeneration = 0;
const flowColor = "#0bb151";
const graphViewLabels = { nodes: "ROS 节点层级", components: "组件层级", tracepoints: "埋点层级" } as const;

async function createElk() {
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  return new ELK();
}

const shownComponents = computed(() => {
  if (!topology.value) return [];
  const all = topology.value.nodes;
  const byId = new Map(all.map((component) => [component.component_id, component]));
  let allowed = new Set(all.map((component) => component.component_id));
  if (componentId.value) {
    allowed = new Set([componentId.value]);
    let changed = true;
    while (changed) {
      changed = false;
      all.forEach((component) => {
        if (allowed.has(component.parent_id) && !allowed.has(component.component_id)) {
          allowed.add(component.component_id);
          changed = true;
        }
      });
    }
    let parent = byId.get(componentId.value)?.parent_id;
    while (parent) {
      allowed.add(parent);
      parent = byId.get(parent)?.parent_id;
    }
    topology.value.edges.forEach((edge) => {
      if (edge.source_id === componentId.value) allowed.add(edge.target_id);
      if (edge.target_id === componentId.value) allowed.add(edge.source_id);
    });
  }
  const term = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  if (term) {
    const matches = new Set(
      all
        .filter((component) => `${component.name} ${component.component_id} ${component.kind}`.toLocaleLowerCase("zh-CN").includes(term))
        .map((component) => component.component_id),
    );
    for (const id of [...matches]) {
      let parent = byId.get(id)?.parent_id;
      while (parent) {
        matches.add(parent);
        parent = byId.get(parent)?.parent_id;
      }
    }
    allowed = new Set([...allowed].filter((id) => matches.has(id)));
  }
  return all.filter((component) => allowed.has(component.component_id));
});

async function layoutGraph(): Promise<void> {
  const generation = ++layoutGeneration;
  const topologyValue = topology.value;
  const components = shownComponents.value;
  if (!topologyValue || !components.length) {
    nodes.value = [];
    edges.value = [];
    layoutPending.value = false;
    return;
  }
  layoutPending.value = true;
  const shown = new Set(components.map((component) => component.component_id));
  const graphEdges = topologyValue.edges.filter((edge) => shown.has(edge.source_id) && shown.has(edge.target_id));
  try {
    const elk = await (elkPromise ??= createElk());
    const result = await elk.layout({
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.spacing.nodeNode": "46",
        "elk.layered.spacing.nodeNodeBetweenLayers": "196",
        "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      },
      children: components.map((component) => ({ id: component.component_id, width: 206, height: 88 })),
      edges: graphEdges.map((edge) => ({ id: edge.edge_id, sources: [edge.source_id], targets: [edge.target_id] })),
    });
    if (generation !== layoutGeneration) return;
    const byId = new Map(topologyValue.nodes.map((component) => [component.component_id, component]));
    const nextNodes: Node[] = (result.children ?? []).map((child) => {
      const component = byId.get(child.id)!;
      return {
        id: component.component_id,
        type: "topology",
        position: { x: child.x ?? 0, y: child.y ?? 0 },
        data: {
          label: component.name,
          componentId: component.component_id,
          kind: component.kind,
          parentName: component.parent_id ? byId.get(component.parent_id)?.name ?? component.parent_id : "",
          provenance: component.provenance,
          processingMs: component.metric_value_ms,
          scoped: component.component_id === componentId.value,
        },
      };
    });
    nodes.value = nextNodes;
    const nextEdges: Edge[] = graphEdges.map((edge) => {
      const contains = edge.kind === "contains";
      return {
        id: edge.edge_id,
        source: edge.source_id,
        target: edge.target_id,
        type: "topology",
        markerEnd: edge.directed ? { type: MarkerType.ArrowClosed, color: flowColor } : undefined,
        animated: !contains,
        class: contains ? "contains-edge" : "flow-edge",
        data: {
          ...edge,
          displayLabel: contains ? "" : edge.name,
          displayMetric: !contains && edge.metric_value_ms !== null ? formatDuration(edge.metric_value_ms) : "",
        },
      };
    });
    if (generation !== layoutGeneration) return;
    edges.value = nextEdges;
    await nextTick();
    if (generation === layoutGeneration) {
      window.requestAnimationFrame(() => {
        if (generation === layoutGeneration) fitView({ padding: 0.16, duration: 220 });
      });
    }
  } finally {
    if (generation === layoutGeneration) layoutPending.value = false;
  }
}

function onNodeClick(event: NodeMouseEvent): void {
  store.selectComponent(event.node.id);
}

function onEdgeClick(event: { edge: Edge }): void {
  const edge = topology.value?.edges.find((item) => item.edge_id === event.edge.id);
  if (edge) store.selection = { kind: "edge", title: edge.name, subtitle: edge.edge_id, data: recordToData(edge) };
}

watch([topology, shownComponents, requestId, metric], layoutGraph, { immediate: true });
</script>

<template>
  <section class="topology-panel" :class="{ 'motion-paused': motionPaused }" aria-label="调用拓扑">
    <header class="panel-heading">
      <div>
        <strong>调用拓扑</strong>
        <span v-if="topology">{{ topology.topology_source === "declared" ? "清单拓扑" : "观测拓扑" }} · {{ graphViewLabels[topology.view] }}</span>
      </div>
      <span v-if="requestId" class="scope-indicator">单请求实际值</span>
      <span v-else class="scope-indicator">{{ metricLabel(metric) }} 聚合</span>
      <button class="icon-button" type="button" title="适配全部节点" :disabled="!nodes.length" @click="fitView({ padding: 0.16, duration: 220 })">
        <AppIcon name="fit" /><span class="sr-only">适配全部节点</span>
      </button>
    </header>
    <div class="topology-canvas">
      <AsyncState
        :state="analysisState"
        :error="analysisError"
        :empty="!layoutPending && !nodes.length"
        empty-title="没有可显示的拓扑"
        empty-message="当前追踪可能缺少拓扑清单，或过滤条件未命中节点。"
        @retry="analysisId && store.openAnalysis(analysisId)"
      >
        <VueFlow
          id="ibrobot-topology"
          v-model:nodes="nodes"
          v-model:edges="edges"
          :nodes-draggable="true"
          :nodes-connectable="false"
          :elements-selectable="true"
          :min-zoom="0.32"
          :max-zoom="2.2"
          :default-viewport="{ x: 20, y: 20, zoom: 0.8 }"
          fit-view-on-init
          @node-click="onNodeClick"
          @edge-click="onEdgeClick"
        >
          <Background pattern-color="rgba(0,0,0,.10)" :gap="24" :size="1" />
          <Panel class="topology-legend" position="top-right" role="note" aria-label="拓扑关系图例">
            <strong>关系图例</strong>
            <span><i class="directed" aria-hidden="true" /><span><b>有向箭头</b> 调用或数据流方向</span></span>
            <span><i class="contains" aria-hidden="true" /><span><b>灰色虚线</b> 层级包含关系</span></span>
          </Panel>
          <Controls position="bottom-left" :show-interactive="false" />
          <MiniMap position="bottom-right" :pannable="true" :zoomable="true" node-color="#5177CA" mask-color="rgba(243,243,245,.78)" />
          <template #node-topology="props"><TopologyNode v-bind="props" /></template>
          <template #edge-topology="props"><TopologyEdge v-bind="props" /></template>
        </VueFlow>
      </AsyncState>
    </div>
  </section>
</template>
