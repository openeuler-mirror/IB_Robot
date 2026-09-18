<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import type { DetailTab } from "../api/types";
import { useProfilerStore } from "../stores/profiler";
import InspectorPanel from "./InspectorPanel.vue";
import ComparisonPanel from "./tabs/ComparisonPanel.vue";
import CriticalPathPanel from "./tabs/CriticalPathPanel.vue";
import DistributionPanel from "./tabs/DistributionPanel.vue";
import EventsPanel from "./tabs/EventsPanel.vue";
import FlowsPanel from "./tabs/FlowsPanel.vue";
import RequestsPanel from "./tabs/RequestsPanel.vue";
import SpansPanel from "./tabs/SpansPanel.vue";
import SummaryPanel from "./tabs/SummaryPanel.vue";
import SpanProfilePanel from "./tabs/SpanProfilePanel.vue";
import TimelinePanel from "./tabs/TimelinePanel.vue";
import WarningsPanel from "./tabs/WarningsPanel.vue";

const store = useProfilerStore();
const { activeTab, capabilities, summary, distribution, comparisonResult, timeline, callTree, spanAnalysisMode, spanProfile, criticalPath, events, spans, flows, warnings, selection } = storeToRefs(store);
const allTabs: { id: DetailTab; label: string; count?: () => number }[] = [
  { id: "summary", label: "汇总" },
  { id: "requests", label: "请求", count: () => summary.value?.analysis.request_count ?? 0 },
  { id: "distribution", label: "分布", count: () => distribution.value?.sample_count ?? 0 },
  { id: "comparison", label: "基线比较", count: () => comparisonResult.value?.metrics.length ?? 0 },
  { id: "timeline", label: "时间线", count: () => timeline.value?.items.length ?? 0 },
  { id: "span-profile", label: "Span 分析", count: () => spanAnalysisMode.value === "call-tree" ? callTree.value?.nodes.length ?? 0 : spanProfile.value?.returned_nodes ?? 0 },
  { id: "critical-path", label: "关键路径", count: () => criticalPath.value?.returned_segments ?? 0 },
  { id: "events", label: "事件", count: () => events.value.length },
  { id: "spans", label: "Span", count: () => spans.value.length },
  { id: "flows", label: "Flow", count: () => flows.value.length },
  { id: "warnings", label: "警告", count: () => warnings.value.length },
];
const tabs = computed(() => allTabs.filter((tab) => (
  tab.id !== "comparison"
  || capabilities.value?.baseline_compare
  || (!capabilities.value && activeTab.value === "comparison")
)));
const activeIndex = computed(() => tabs.value.findIndex((tab) => tab.id === activeTab.value));

function selectTab(tab: DetailTab): void {
  activeTab.value = tab;
  store.fetchTab(tab);
}

function moveTab(delta: number): void {
  const next = (activeIndex.value + delta + tabs.value.length) % tabs.value.length;
  selectTab(tabs.value[next].id);
  document.getElementById(`detail-tab-${tabs.value[next].id}`)?.focus();
}
</script>

<template>
  <section class="detail-pane" aria-label="追踪详情">
    <nav class="detail-tabs" role="tablist" aria-label="分析视图" @keydown.left.prevent="moveTab(-1)" @keydown.right.prevent="moveTab(1)">
      <button
        v-for="tab in tabs"
        :id="`detail-tab-${tab.id}`"
        :key="tab.id"
        type="button"
        role="tab"
        :aria-selected="activeTab === tab.id"
        :tabindex="activeTab === tab.id ? 0 : -1"
        :class="{ active: activeTab === tab.id }"
        @click="selectTab(tab.id)"
      >
        {{ tab.label }}<span v-if="tab.count && tab.count()">{{ tab.count!() }}</span>
      </button>
    </nav>
    <div class="detail-workarea" :class="{ 'with-inspector': selection }">
      <div class="detail-panel" role="tabpanel" :aria-labelledby="`detail-tab-${activeTab}`">
        <SummaryPanel v-if="activeTab === 'summary'" />
        <RequestsPanel v-else-if="activeTab === 'requests'" />
        <DistributionPanel v-else-if="activeTab === 'distribution'" />
        <ComparisonPanel v-else-if="activeTab === 'comparison'" />
        <TimelinePanel v-else-if="activeTab === 'timeline'" />
        <SpanProfilePanel v-else-if="activeTab === 'span-profile'" />
        <CriticalPathPanel v-else-if="activeTab === 'critical-path'" />
        <EventsPanel v-else-if="activeTab === 'events'" />
        <SpansPanel v-else-if="activeTab === 'spans'" />
        <FlowsPanel v-else-if="activeTab === 'flows'" />
        <WarningsPanel v-else />
      </div>
      <InspectorPanel />
    </div>
  </section>
</template>
