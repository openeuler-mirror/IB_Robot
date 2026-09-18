<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, formatInteger, formatTimestamp, recordToData, stageLabel } from "../../utils/format";
import AppIcon from "../AppIcon.vue";
import AsyncState from "../AsyncState.vue";
import DataGrid, { type GridColumn } from "../DataGrid.vue";

const store = useProfilerStore();
const { summary, tabState, tabError, searchQuery, analysisId } = storeToRefs(store);

const stages = computed(() => {
  const rows = Object.entries(summary.value?.stages ?? {});
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return query ? rows.filter(([name]) => `${name} ${stageLabel(name)}`.toLocaleLowerCase("zh-CN").includes(query)) : rows;
});

const spanColumns: GridColumn[] = [
  { key: "name", label: "Span" },
  { key: "component_id", label: "组件", className: "mono-cell" },
  { key: "origin", label: "来源" },
  { key: "status", label: "记录状态" },
  ...["p50", "p95", "maximum", "mean"].map((key): GridColumn => ({
    key, label: key === "maximum" ? "max" : key, align: "right", className: "numeric",
    format: (value) => formatDuration(value as number | null),
  })),
  { key: "count", label: "Occurrence 数", align: "right", className: "numeric" },
];

const spanRows = computed(() => {
  const rows = summary.value?.span_summary ?? [];
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return rows
    .filter((item) => `${item.name} ${item.component_id} ${item.origin} ${item.status}`.toLocaleLowerCase("zh-CN").includes(query))
    .map((item) => ({ ...recordToData(item), id: JSON.stringify([item.component_id, item.name, item.origin, item.status]) }));
});

async function openSpanAnalysis(): Promise<void> {
  store.componentId = "";
  store.searchQuery = "";
  await store.setSpanAnalysisMode("aggregate");
  store.activeTab = "span-profile";
  await store.fetchTab("span-profile");
}
</script>

<template>
  <AsyncState
    :state="tabState.summary"
    :error="tabError.summary"
    :empty="!summary"
    empty-title="尚未载入分析"
    empty-message="从页首选择追踪源并载入。"
    @retry="analysisId && store.openAnalysis(analysisId)"
  >
    <div v-if="summary" class="tab-scroll summary-view">
      <header class="profile-semantic-banner">
        <strong>请求链投影</strong>
        <span>阶段延迟仅汇总可唯一关联的请求指标，不代表业务成功或可用性。歧义、错误、不完整记录不纳入该投影，不等于未采到，也不表示需要重试。</span>
      </header>
      <section class="summary-strip">
        <dl><dt>请求</dt><dd>{{ formatInteger(summary.analysis.request_count) }}</dd></dl>
        <dl><dt>事件</dt><dd>{{ formatInteger(summary.analysis.event_count) }}</dd></dl>
        <dl><dt>Span 记录</dt><dd>{{ formatInteger(summary.analysis.span_count) }}</dd></dl>
        <dl><dt>关联 Flow</dt><dd>{{ formatInteger(summary.analysis.flow_count) }}</dd></dl>
        <dl>
          <dt>
            投影指标覆盖
            <span
              class="term-help"
              tabindex="0"
              role="img"
              aria-label="投影指标覆盖说明：有数值样本的预期指标数除以预期指标总数；未形成投影可能是未采到、歧义、错误或不完整，不代表业务成功率。"
              title="有数值样本的预期指标数 / 预期指标总数；未形成投影可能是未采到、歧义、错误或不完整，不代表业务成功率。"
            ><AppIcon name="help" :size="11" /></span>
          </dt>
          <dd>{{ summary.coverage.observed_metrics ?? 0 }} / {{ summary.coverage.expected_metrics ?? 0 }}</dd>
        </dl>
        <dl class="generated"><dt>生成时间</dt><dd>{{ formatTimestamp(summary.analysis.created_at) }}</dd></dl>
      </section>

      <section class="matrix-section">
        <header><strong>请求链投影延迟矩阵</strong><span>{{ summary.coverage.boundary || "分发请求至首次动作执行" }} · 主机侧打点耗时，不等同设备执行时间</span></header>
        <div class="data-grid-wrap">
          <table class="data-grid summary-matrix">
            <thead><tr><th>阶段</th><th class="align-right">min</th><th class="align-right">p50</th><th class="align-right">p95</th><th class="align-right">p99</th><th class="align-right">max</th><th class="align-right">mean</th><th class="align-right">样本</th></tr></thead>
            <tbody>
              <tr v-if="!stages.length"><td colspan="8" class="empty-cell">没有匹配的投影数值样本；请在请求视图查看指标状态，或在 Span 分析查看记录。</td></tr>
              <tr v-for="([name, stats]) in stages" :key="name" :class="{ emphasized: name === 'total_ms' }">
                <td><strong>{{ stageLabel(name) }}</strong><code>{{ name }}</code></td>
                <td class="align-right numeric">{{ formatDuration(stats.minimum) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.p50) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.p95) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.p99) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.maximum) }}</td>
                <td class="align-right numeric">{{ formatDuration(stats.mean) }}</td>
                <td class="align-right numeric">{{ formatInteger(stats.count) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section class="matrix-section">
        <header><strong>通用 Span 汇总</strong><button class="text-button" type="button" @click="openSpanAnalysis">更多见Span分析</button></header>
        <p class="grid-limit">按 component / name / origin / status 分组，统计全部完整且耗时有效的 occurrence，包含错误及重复调用，不挑选最早成功记录；不完整记录见 Span 分析。</p>
        <div class="grid-limit">
          {{ summary.span_summary_truncated ? "已截断，返回前" : "返回" }} {{ summary.span_summary.length }} / {{ summary.span_summary_total }} 组，上限 {{ summary.span_summary_limit }} 组；搜索仅作用于此预览。
        </div>
        <DataGrid :rows="spanRows" :columns="spanColumns" row-key="id" empty-text="当前预览没有匹配的完整 Span 分组" />
      </section>

      <section v-if="summary.custom_mark_summary.length" class="mark-section">
        <header><strong>用户事件计数</strong></header>
        <button v-for="mark in summary.custom_mark_summary" :key="`${mark.component_id}:${mark.name}`" type="button" @click="store.componentId = mark.component_id">
          <span>{{ mark.name }}</span><code>{{ mark.component_id || "未绑定组件" }}</code><b>{{ mark.count }}</b>
        </button>
      </section>

      <section v-if="summary.coverage.missing_events?.length" class="coverage-warning">
        <strong>未形成投影数值</strong>
        <span>{{ summary.coverage.missing_events.map(stageLabel).join("、") }}。不等于事件未采到；请在请求视图查看 *_status 区分歧义、错误、不完整与未采到。</span>
      </section>
    </div>
  </AsyncState>
</template>
