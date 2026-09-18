<script setup lang="ts">
import { computed, ref } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordToData } from "../../utils/format";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { requests, requestId, tabState, tabError, searchQuery, analysisId } = storeToRefs(store);
const sortBy = ref("total_ms");
const ascending = ref(false);

const columns: GridColumn[] = [
  { key: "request_id", label: "请求标识", width: "20%", className: "mono-cell" },
  ...[
    ["total_ms", "端到端", "12%"],
    ["inference_ms", "模型调用", "12%"],
    ["preprocess_ms", "预处理", "11%"],
    ["postprocess_ms", "后处理", "11%"],
    ["queue_refill_ms", "队列补充", "12%"],
  ].map(([key, label, width]): GridColumn => ({
    key, label, width, align: "right", className: "numeric",
    format: (value, row) => {
      const status = String(row[`${key}_status`] ?? "");
      if (status && status !== "ok") {
        const labels: Record<string, string> = { ambiguous: "歧义", error: "错误记录", incomplete: "记录不完整", invalid: "无效数据" };
        return status.split(",").map((part) => labels[part] ? `${labels[part]} (${part})` : part).join(" / ");
      }
      return typeof value === "number" && Number.isFinite(value) ? formatDuration(value) : "未采到";
    },
  })),
  { key: "total_ms_source", label: "端到端来源", width: "11%" },
  { key: "inference_ms_source", label: "推理来源", width: "11%" },
  { key: "actions", label: "打开请求视图", width: "180px", className: "request-actions" },
];

const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  const filtered = query
    ? requests.value.filter((request) => Object.values(request).some((value) => String(value ?? "").toLocaleLowerCase("zh-CN").includes(query)))
    : [...requests.value];
  return filtered
    .sort((a, b) => {
      const left = Number(a[sortBy.value] ?? Number.NEGATIVE_INFINITY);
      const right = Number(b[sortBy.value] ?? Number.NEGATIVE_INFINITY);
      return (left - right) * (ascending.value ? 1 : -1);
    })
    .map((row) => recordToData(row));
});

function select(row: Record<string, unknown>): void {
  store.selectRequest(String(row.request_id));
}

function openRequest(id: string, tab: "timeline" | "span-profile" | "critical-path"): void {
  void store.navigateToRequest(id, tab);
}
</script>

<template>
  <AsyncState :state="tabState.requests" :error="tabError.requests" :empty="!requests.length" empty-title="没有请求记录" empty-message="追踪中没有可关联的 request_id。" @retry="analysisId && store.openAnalysis(analysisId)">
    <div class="table-panel">
      <div class="table-toolbar">
        <span title="缺值优先显示 *_status：歧义、错误或不完整；无状态且无数值才显示未采到。这些是观测状态，不判定业务可用性，也不建议重试。">请求链投影 · 共 {{ rows.length }} 条请求</span>
        <label>排序<select v-model="sortBy"><option value="total_ms">端到端</option><option value="inference_ms">模型调用</option><option value="preprocess_ms">预处理</option><option value="queue_refill_ms">队列补充</option></select></label>
        <button class="text-button" type="button" @click="ascending = !ascending">{{ ascending ? "升序" : "降序" }}</button>
        <span class="toolbar-help" title="单击设置全局请求范围，J / K 移动，回车检查">状态仅反映观测；全部 occurrence 见 Span 分析</span>
      </div>
      <DataGrid :rows="rows" :columns="columns" row-key="request_id" :selected-key="requestId" @select="select">
        <template #cell-actions="{ row }">
          <button type="button" class="text-button" :aria-label="`在时间线打开请求 ${row.request_id}`" @click.stop="openRequest(String(row.request_id), 'timeline')">时间线</button>
          <button type="button" class="text-button" :aria-label="`在 Span 分析打开请求 ${row.request_id}`" @click.stop="openRequest(String(row.request_id), 'span-profile')">Span 分析</button>
          <button type="button" class="text-button" :aria-label="`在关键路径打开请求 ${row.request_id}`" @click.stop="openRequest(String(row.request_id), 'critical-path')">关键路径</button>
        </template>
      </DataGrid>
    </div>
  </AsyncState>
</template>
