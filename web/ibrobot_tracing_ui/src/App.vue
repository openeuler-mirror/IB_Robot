<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useRoute, useRouter } from "vue-router";
import DetailPane from "./components/DetailPane.vue";
import ProfilerHeader from "./components/ProfilerHeader.vue";
import ScopeBar from "./components/ScopeBar.vue";
import TopologyGraph from "./components/TopologyGraph.vue";
import { useProfilerStore } from "./stores/profiler";
import { formatInteger, shortId } from "./utils/format";
import { buildProfilerQuery, DETAIL_TABS, parseProfilerRoute } from "./utils/routeState";

const store = useProfilerStore();
const route = useRoute();
const router = useRouter();
const {
  activeTab,
  analysisId,
  analysisState,
  capabilities,
  summary,
  sourceId,
  requestId,
  componentId,
  distributionMetric,
  metric,
  graphView,
  spanProfile,
  criticalPath,
  spanAnalysisMode,
  searchQuery,
  selection,
  selectedBaselineSourceId,
  comparisonStatistic,
  comparisonRelativeThreshold,
  comparisonAbsoluteThresholdMs,
  comparisonMetric,
  comparisonJob,
  comparisonResult,
} = storeToRefs(store);
const topPercent = ref(48);
const resizing = ref(false);
const workspace = ref<HTMLElement | null>(null);
let routeApplying = false;

const statusText = computed(() => {
  if (analysisState.value === "loading") return "正在解析追踪";
  if (analysisState.value === "error") return "分析不可用";
  if (analysisState.value === "ready") return "分析就绪";
  return "等待载入追踪";
});

async function applyRoute(): Promise<void> {
  routeApplying = true;
  const state = parseProfilerRoute(route.query);
  activeTab.value = state.activeTab;
  requestId.value = state.requestId;
  componentId.value = state.componentId;
  distributionMetric.value = state.distributionMetric;
  metric.value = state.metric;
  graphView.value = state.graphView;
  await store.setSpanAnalysisMode(state.spanAnalysisMode);
  searchQuery.value = state.searchQuery;
  selectedBaselineSourceId.value = state.selectedBaselineSourceId;
  comparisonStatistic.value = state.comparisonStatistic;
  comparisonRelativeThreshold.value = state.comparisonRelativeThreshold;
  comparisonAbsoluteThresholdMs.value = state.comparisonAbsoluteThresholdMs;
  comparisonMetric.value = state.comparisonMetric;
  const routeAnalysisId = Array.isArray(route.params.analysisId)
    ? String(route.params.analysisId[0] ?? "")
    : String(route.params.analysisId ?? "");
  if (routeAnalysisId && routeAnalysisId !== analysisId.value) await store.openAnalysis(routeAnalysisId);
  if (routeAnalysisId) await store.fetchTab(activeTab.value);
  await nextTick();
  routeApplying = false;
}

async function syncRoute(): Promise<void> {
  if (routeApplying) return;
  const query = buildProfilerQuery({
    activeTab: activeTab.value,
    requestId: requestId.value,
    componentId: componentId.value,
    distributionMetric: distributionMetric.value,
    metric: metric.value,
    graphView: graphView.value,
    spanAnalysisMode: spanAnalysisMode.value,
    searchQuery: searchQuery.value,
    selectedBaselineSourceId: selectedBaselineSourceId.value,
    comparisonStatistic: comparisonStatistic.value,
    comparisonRelativeThreshold: comparisonRelativeThreshold.value,
    comparisonAbsoluteThresholdMs: comparisonAbsoluteThresholdMs.value,
    comparisonMetric: comparisonMetric.value,
  });
  const params = analysisId.value ? { analysisId: analysisId.value } : {};
  await router.replace({ name: "analysis", params, query });
}

async function analyze(): Promise<void> {
  try {
    const id = await store.analyzeSelectedSource();
    analysisId.value = id;
    await syncRoute();
  } catch (error) {
    if (!(error instanceof DOMException && error.name === "AbortError")) console.error(error);
  }
}

function beginResize(event: PointerEvent): void {
  if (!workspace.value || window.matchMedia("(max-width: 900px)").matches) return;
  resizing.value = true;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function resize(event: PointerEvent): void {
  if (!resizing.value || !workspace.value) return;
  const bounds = workspace.value.getBoundingClientRect();
  topPercent.value = Math.min(70, Math.max(30, ((event.clientY - bounds.top) / bounds.height) * 100));
}

function endResize(): void {
  resizing.value = false;
}

function handleKeyboard(event: KeyboardEvent): void {
  const target = event.target as HTMLElement;
  const editing = target.matches("input, select, textarea, [contenteditable='true']");
  if (event.key === "/" && !editing) {
    event.preventDefault();
    document.getElementById("global-trace-search")?.focus();
  } else if (event.key === "Escape") {
    if (selection.value) store.selection = null;
    else if (searchQuery.value) searchQuery.value = "";
  } else if (event.altKey && /^[1-9]$/.test(event.key)) {
    event.preventDefault();
    const tabs = DETAIL_TABS.filter((tab) => tab !== "comparison" || capabilities.value?.baseline_compare);
    const tab = tabs[Number(event.key) - 1];
    if (!tab) return;
    activeTab.value = tab;
    store.fetchTab(activeTab.value);
  }
}

watch(() => route.fullPath, applyRoute, { immediate: true });
watch([
  activeTab,
  analysisId,
  requestId,
  componentId,
  distributionMetric,
  metric,
  graphView,
  spanAnalysisMode,
  searchQuery,
  selectedBaselineSourceId,
  comparisonStatistic,
  comparisonRelativeThreshold,
  comparisonAbsoluteThresholdMs,
  comparisonMetric,
], syncRoute);
watch(capabilities, (value) => {
  if (value && !value.baseline_compare && activeTab.value === "comparison") activeTab.value = "summary";
});
watch(graphView, () => {
  if (routeApplying) return;
  componentId.value = "";
  if (selection.value?.kind === "component" || selection.value?.kind === "edge") selection.value = null;
});
watch([metric, graphView, requestId], () => store.refreshTopology());
watch([requestId, componentId], () => {
  if (!routeApplying) void store.refreshScopedData();
});
watch(sourceId, (value) => {
  if (summary.value && value !== summary.value.analysis.source_id) store.clearCriticalPath();
});

onMounted(() => {
  store.loadSources();
  window.addEventListener("keydown", handleKeyboard);
});
onBeforeUnmount(() => window.removeEventListener("keydown", handleKeyboard));
</script>

<template>
  <div class="profiler-app" :class="{ resizing }">
    <ProfilerHeader @analyze="analyze" @refresh="store.refreshSources" />
    <ScopeBar />
    <main ref="workspace" class="workspace" :style="{ '--top-pane': `${topPercent}%` }">
      <TopologyGraph />
      <div
        class="pane-resizer"
        role="separator"
        aria-label="调整拓扑与详情高度"
        aria-orientation="horizontal"
        :aria-valuenow="Math.round(topPercent)"
        tabindex="0"
        @pointerdown="beginResize"
        @pointermove="resize"
        @pointerup="endResize"
        @pointercancel="endResize"
        @keydown.up.prevent="topPercent = Math.max(30, topPercent - 2)"
        @keydown.down.prevent="topPercent = Math.min(70, topPercent + 2)"
      ><i /></div>
      <DetailPane />
    </main>
    <footer class="status-bar">
      <span class="status-state"><i :class="analysisState" />{{ statusText }}</span>
      <span v-if="summary">{{ formatInteger(summary.analysis.request_count) }} 个请求 · {{ formatInteger(summary.analysis.event_count) }} 个事件</span>
      <span v-if="requestId">请求 {{ shortId(requestId) }}</span>
      <span v-if="componentId">组件 {{ componentId }}</span>
      <span v-if="activeTab === 'span-profile'">{{ spanAnalysisMode === "call-tree" ? "调用树" : spanAnalysisMode === "request" ? "请求 Span 区间" : "聚合 Span 剖析" }} · 冰柱</span>
      <span v-if="activeTab === 'span-profile' && spanProfile">{{ formatInteger(spanProfile.returned_nodes) }}/{{ formatInteger(spanProfile.total_nodes) }} 个节点<span v-if="spanProfile.truncated">（已截断）</span></span>
      <span v-if="activeTab === 'critical-path' && criticalPath">{{ formatInteger(criticalPath.returned_segments) }}/{{ formatInteger(criticalPath.total_segments) }} 个墙钟分段<span v-if="criticalPath.truncated">（返回前缀）</span></span>
      <span v-if="activeTab === 'comparison' && comparisonJob">比较任务 {{ comparisonJob.status }}</span>
      <span v-if="activeTab === 'comparison' && comparisonResult">{{ comparisonResult.comparable ? "可比较" : "不可比较" }} · {{ comparisonResult.has_regression ? "发现回归" : "未发现回归" }}</span>
      <span v-if="capabilities?.authentication === 'none'" class="security-warning">局域网无认证模式</span>
      <span class="status-spacer" />
      <span class="keyboard-hint">/ 过滤 · Esc 关闭检查器 · Alt+1…9 切换视图</span>
    </footer>
  </div>
</template>
