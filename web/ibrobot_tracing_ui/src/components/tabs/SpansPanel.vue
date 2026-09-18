<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { spans, tabState, tabError, searchQuery } = storeToRefs(store);
const columns: GridColumn[] = [
  { key: "name", label: "Span", width: "16%", className: "mono-cell" },
  { key: "component_id", label: "组件", width: "19%", className: "mono-cell" },
  { key: "span_id", label: "Span 标识", width: "13%", className: "mono-cell" },
  { key: "parent_span_id", label: "父标识", width: "13%", className: "mono-cell" },
  { key: "duration_ms", label: "耗时", width: "11%", align: "right", className: "numeric", format: (value) => formatDuration(value as number | null, 3) },
  { key: "origin", label: "来源", width: "8%", format: (value) => value === "user" ? "用户" : "内置" },
  { key: "status", label: "状态", width: "8%" },
  { key: "trace_id", label: "请求标识", width: "12%", className: "mono-cell" },
];
const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return spans.value
    .filter((span) => !query || `${span.name} ${span.component_id} ${span.span_id} ${span.status}`.toLocaleLowerCase("zh-CN").includes(query))
    .map(recordToData);
});
function select(row: Record<string, unknown>): void {
  store.selection = { kind: "span", title: String(row.name), subtitle: String(row.component_id || ""), data: row };
}
</script>

<template>
  <AsyncState :state="tabState.spans" :error="tabError.spans" :empty="!spans.length" empty-title="没有配对 Span" empty-message="当前范围没有完整或未完成的结构化 Span。" @retry="store.fetchTab('spans', true)">
    <div class="table-panel"><div class="table-toolbar"><span>共 {{ rows.length }} 条 Span</span><span class="toolbar-help">单击记录查看 fields 与边界时间</span></div><DataGrid :rows="rows" :columns="columns" row-key="id" @select="select" /></div>
  </AsyncState>
</template>
