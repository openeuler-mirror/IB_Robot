<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { eventComponent, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { events, tabState, tabError, searchQuery } = storeToRefs(store);
const columns: GridColumn[] = [
  { key: "timestamp_ns", label: "时间戳（纳秒）", width: "18%", align: "right", className: "numeric" },
  { key: "name", label: "事件", width: "18%", className: "mono-cell" },
  { key: "component", label: "组件", width: "18%", className: "mono-cell" },
  { key: "provider", label: "提供者", width: "14%", className: "mono-cell" },
  { key: "schema_version", label: "模式", width: "7%", align: "right", className: "numeric" },
  { key: "fields", label: "字段", width: "25%", className: "mono-cell", format: (value) => JSON.stringify(value) },
];

const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return events.value
    .filter((event) => !query || `${event.name} ${eventComponent(event)} ${JSON.stringify(event.fields)}`.toLocaleLowerCase("zh-CN").includes(query))
    .map((event) => ({ ...recordToData(event), component: eventComponent(event), provider: event.origin.provider }));
});

function select(row: Record<string, unknown>): void {
  store.selection = { kind: "event", title: String(row.name), subtitle: String(row.component), data: row };
}
</script>

<template>
  <AsyncState :state="tabState.events" :error="tabError.events" :empty="!events.length" empty-title="没有原始事件" empty-message="当前请求和组件范围内没有事件。" @retry="store.fetchTab('events', true)">
    <div class="table-panel"><div class="table-toolbar"><span>共 {{ rows.length }} 条事件</span><span class="toolbar-help">单页最多读取 1000 条；使用范围栏缩小结果</span></div><DataGrid :rows="rows" :columns="columns" row-key="id" @select="select" /></div>
  </AsyncState>
</template>
