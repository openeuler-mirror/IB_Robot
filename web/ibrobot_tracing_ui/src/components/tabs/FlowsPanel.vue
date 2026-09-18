<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { flows, tabState, tabError, searchQuery } = storeToRefs(store);
const columns: GridColumn[] = [
  { key: "edge_id", label: "边标识", width: "21%", className: "mono-cell" },
  { key: "flow_id", label: "Flow 标识", width: "17%", className: "mono-cell" },
  { key: "trace_id", label: "请求标识", width: "17%", className: "mono-cell" },
  { key: "send_ns", label: "发送（纳秒）", width: "16%", align: "right", className: "numeric" },
  { key: "receive_ns", label: "接收（纳秒）", width: "16%", align: "right", className: "numeric" },
  { key: "duration_ms", label: "耗时", width: "8%", align: "right", className: "numeric", format: (value) => formatDuration(value as number | null, 3) },
  { key: "status", label: "状态", width: "7%", format: (value) => value === "complete" ? "完整" : value === "incomplete" ? "不完整" : value === "negative" ? "负耗时" : String(value) },
];
const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return flows.value
    .filter((flow) => !query || `${flow.edge_id} ${flow.flow_id} ${flow.trace_id} ${flow.status}`.toLocaleLowerCase("zh-CN").includes(query))
    .map(recordToData);
});
function select(row: Record<string, unknown>): void {
  store.selection = { kind: "flow", title: String(row.edge_id), subtitle: String(row.flow_id), data: row };
}
</script>

<template>
  <AsyncState :state="tabState.flows" :error="tabError.flows" :empty="!flows.length" empty-title="没有关联 Flow" empty-message="发送端与接收端需要使用相同的请求、边和 Flow 标识。" @retry="store.fetchTab('flows', true)">
    <div class="table-panel"><div class="table-toolbar"><span>共 {{ rows.length }} 条 Flow</span><span class="toolbar-help">负耗时通常表示跨主机时钟未同步</span></div><DataGrid :rows="rows" :columns="columns" row-key="id" @select="select" /></div>
  </AsyncState>
</template>
