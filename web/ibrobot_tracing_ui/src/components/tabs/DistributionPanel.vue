<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import type { LatencyDistributionBucket } from "../../api/types";
import { useProfilerStore } from "../../stores/profiler";
import { buildHistogramGeometry, valuePosition } from "../../utils/distribution";
import { formatDuration, formatInteger, shortId, stageLabel } from "../../utils/format";
import AsyncState from "../AsyncState.vue";

const store = useProfilerStore();
const { analysisId, distribution, distributionMetric, tabError, tabState } = storeToRefs(store);
const chartHost = ref<HTMLElement | null>(null);
const chartWidth = ref(640);
const hoveredBucketIndex = ref<number | null>(null);
const selectedBucketIndex = ref<number | null>(null);
const plotHeight = 126;
const plotInset = 12;
let resizeObserver: ResizeObserver | null = null;

const metricOptions = computed(() => Array.from(new Set([
  distributionMetric.value,
  ...(distribution.value?.available_metrics ?? []),
].filter(Boolean))));
const plotWidth = computed(() => Math.max(1, chartWidth.value - plotInset * 2));
const geometry = computed(() => buildHistogramGeometry(distribution.value?.buckets ?? [], plotWidth.value, plotHeight, 3));
const p95Position = computed(() => {
  const response = distribution.value;
  if (!response || response.summary.p95 === null || response.minimum === null || response.maximum === null) return null;
  return plotInset + valuePosition(response.summary.p95, response.minimum, response.maximum, plotWidth.value);
});
const selectedBucket = computed(() => distribution.value?.buckets.find((bucket) => bucket.index === selectedBucketIndex.value) ?? null);
const hoveredBucket = computed(() => distribution.value?.buckets.find((bucket) => bucket.index === hoveredBucketIndex.value) ?? null);
const hoveredBar = computed(() => geometry.value.bars.find((bar) => bar.index === hoveredBucketIndex.value) ?? null);
const sortedOutliers = computed(() => [...(distribution.value?.outliers ?? [])].sort((left, right) => right.value - left.value || left.rank - right.rank));
const stats = computed(() => {
  const response = distribution.value;
  if (!response) return [];
  return [
    { label: "min", value: formatDuration(response.summary.minimum) },
    { label: "p50", value: formatDuration(response.summary.p50) },
    { label: "p95", value: formatDuration(response.summary.p95) },
    { label: "p99", value: formatDuration(response.summary.p99) },
    { label: "max", value: formatDuration(response.summary.maximum) },
    { label: "mean", value: formatDuration(response.summary.mean) },
    { label: "样本", value: formatInteger(response.sample_count) },
    { label: "无效", value: formatInteger(response.invalid_count) },
    { label: "桶", value: formatInteger(response.bin_count) },
  ];
});

function tailLabel(bucket: LatencyDistributionBucket): string {
  const p95 = distribution.value?.summary.p95;
  if (p95 === null || p95 === undefined) return "无 P95 分类";
  if (bucket.start >= p95) return "P95 尾部区间";
  if (bucket.end < p95) return "P95 以内区间";
  return "跨越 P95 尾部起点";
}

function isTailBucket(bucket: LatencyDistributionBucket): boolean {
  const p95 = distribution.value?.summary.p95;
  return p95 !== null && p95 !== undefined && bucket.end >= p95;
}

function bucketLabel(bucket: LatencyDistributionBucket): string {
  const range = bucket.start === bucket.end
    ? formatDuration(bucket.start)
    : `${formatDuration(bucket.start)} 至 ${formatDuration(bucket.end)}`;
  return `桶 ${bucket.index + 1}，${range}，${bucket.count} 个样本，${tailLabel(bucket)}。按回车或空格选择。`;
}

function selectBucket(index: number): void {
  selectedBucketIndex.value = index;
}

function changeMetric(event: Event): void {
  void store.setDistributionMetric((event.target as HTMLSelectElement).value);
}

function openRequest(requestId: string, tab: "timeline" | "span-profile" | "critical-path"): void {
  void store.navigateToRequest(requestId, tab);
}

function updateChartWidth(): void {
  if (chartHost.value) chartWidth.value = Math.max(280, Math.floor(chartHost.value.getBoundingClientRect().width));
}

watch(distribution, (response) => {
  const largest = response?.buckets.reduce<LatencyDistributionBucket | null>(
    (current, bucket) => current === null || bucket.count > current.count ? bucket : current,
    null,
  );
  selectedBucketIndex.value = largest?.index ?? null;
  hoveredBucketIndex.value = null;
  void nextTick(updateChartWidth);
}, { immediate: true });

watch(chartHost, (element, previous) => {
  if (previous) resizeObserver?.unobserve(previous);
  if (element) {
    resizeObserver?.observe(element);
    updateChartWidth();
  }
});

onMounted(() => {
  resizeObserver = new ResizeObserver(updateChartWidth);
  if (chartHost.value) resizeObserver.observe(chartHost.value);
});

onBeforeUnmount(() => resizeObserver?.disconnect());
</script>

<template>
  <section class="distribution-panel">
    <header class="distribution-toolbar">
      <div>
        <strong>延迟分布</strong>
        <span v-if="distribution?.metric">{{ stageLabel(distribution.metric) }} <code>{{ distribution.metric }}</code></span>
        <span v-else>按请求级延迟生成直方图</span>
      </div>
      <label>
        指标
        <select :value="distributionMetric" aria-label="分布延迟指标" :disabled="tabState.distribution === 'loading' && !distribution" @change="changeMetric">
          <option value="">自动（后端选择）</option>
          <option v-for="name in metricOptions" :key="name" :value="name">{{ stageLabel(name) }} · {{ name }}</option>
        </select>
      </label>
    </header>

    <AsyncState
      class="distribution-async"
      :state="tabState.distribution"
      :error="tabError.distribution"
      :empty="tabState.distribution === 'ready' && (!distribution || !distribution.buckets.length)"
      empty-title="没有可绘制的延迟样本"
      :empty-message="distribution?.metric ? `${stageLabel(distribution.metric)} 没有有效请求值；无效值 ${distribution.invalid_count} 条。` : '当前分析没有请求级延迟指标。'"
      loading-message="正在计算延迟分布…"
      @retry="analysisId && store.fetchTab('distribution', true)"
    >
      <div v-if="distribution" class="distribution-scroll">
        <div class="distribution-stat-strip" aria-label="分布统计">
          <dl v-for="stat in stats" :key="stat.label"><dt>{{ stat.label }}</dt><dd>{{ stat.value }}</dd></dl>
        </div>

        <section class="distribution-chart-section" aria-labelledby="distribution-chart-title">
          <header>
            <div><strong id="distribution-chart-title">请求延迟直方图</strong><span>柱高表示样本数</span></div>
            <span class="tail-legend">P95 尾部区间使用深色柱，并以虚线标出起点</span>
          </header>
          <div ref="chartHost" class="distribution-chart-host">
            <svg
              class="distribution-chart"
              :viewBox="`0 0 ${chartWidth} ${plotHeight + 34}`"
              role="img"
              :aria-label="`${stageLabel(distribution.metric || '')}延迟直方图，共${distribution.sample_count}个样本`"
            >
              <line class="distribution-axis" :x1="plotInset" :x2="chartWidth - plotInset" :y1="plotHeight" :y2="plotHeight" />
              <g
                v-for="bar in geometry.bars"
                :key="bar.index"
                class="distribution-bar"
                :class="{
                  selected: selectedBucketIndex === bar.index,
                  hovered: hoveredBucketIndex === bar.index,
                  tail: isTailBucket(distribution.buckets.find((bucket) => bucket.index === bar.index)!),
                }"
                role="button"
                tabindex="0"
                :aria-label="bucketLabel(distribution.buckets.find((bucket) => bucket.index === bar.index)!)"
                @mouseenter="hoveredBucketIndex = bar.index"
                @mouseleave="hoveredBucketIndex = null"
                @focus="hoveredBucketIndex = bar.index"
                @blur="hoveredBucketIndex = null"
                @click="selectBucket(bar.index)"
                @keydown.enter.prevent="selectBucket(bar.index)"
                @keydown.space.prevent="selectBucket(bar.index)"
              >
                <rect class="distribution-bar-hit" :x="plotInset + bar.x" y="0" :width="bar.width" :height="plotHeight" />
                <rect
                  class="distribution-bar-shape"
                  :x="plotInset + bar.x"
                  :y="bar.y"
                  :width="bar.width"
                  :height="Math.max(bar.height, bar.count ? 1 : 0)"
                />
              </g>
              <g v-if="p95Position !== null" class="distribution-p95-marker" aria-hidden="true">
                <line :x1="p95Position" :x2="p95Position" y1="5" :y2="plotHeight" />
                <text :x="Math.min(p95Position + 4, chartWidth - 74)" y="12">p95 尾部起点</text>
              </g>
              <text class="distribution-axis-label" :x="plotInset" :y="plotHeight + 18">{{ formatDuration(distribution.minimum) }}</text>
              <text v-if="distribution.minimum !== distribution.maximum" class="distribution-axis-label end" :x="chartWidth - plotInset" :y="plotHeight + 18">{{ formatDuration(distribution.maximum) }}</text>
            </svg>
            <div
              v-if="hoveredBucket && hoveredBar"
              class="distribution-tooltip"
              :style="{ left: `${Math.min(94, Math.max(6, (plotInset + hoveredBar.x + hoveredBar.width / 2) * 100 / chartWidth))}%` }"
              role="status"
            >
              <strong>桶 {{ hoveredBucket.index + 1 }} · {{ hoveredBucket.count }} 个样本</strong>
              <span>{{ formatDuration(hoveredBucket.start) }} 至 {{ formatDuration(hoveredBucket.end) }}</span>
              <span>{{ tailLabel(hoveredBucket) }}</span>
            </div>
          </div>
        </section>

        <div class="distribution-detail-grid">
          <section class="distribution-bucket-detail">
            <header>
              <div>
                <strong>选中桶请求</strong>
                <span v-if="selectedBucket">桶 {{ selectedBucket.index + 1 }} · {{ formatDuration(selectedBucket.start) }} 至 {{ formatDuration(selectedBucket.end) }} · {{ tailLabel(selectedBucket) }}</span>
              </div>
              <b v-if="selectedBucket">{{ selectedBucket.count }}</b>
            </header>
            <p v-if="selectedBucket?.truncated" class="distribution-truncation" role="note">
              此桶有 {{ selectedBucket.count }} 个样本，仅返回前 {{ selectedBucket.returned }} 个请求标识。
            </p>
            <div class="distribution-table-wrap">
              <table class="distribution-table">
                <thead><tr><th>请求标识</th><th>打开请求视图</th></tr></thead>
                <tbody>
                  <tr v-if="!selectedBucket?.request_ids.length"><td colspan="2" class="empty-cell">选择有样本的柱以查看请求。</td></tr>
                  <tr v-for="id in selectedBucket?.request_ids" v-else :key="id">
                    <td><code :title="id">{{ shortId(id, 24) }}</code></td>
                    <td class="request-actions">
                      <button type="button" class="text-button" :aria-label="`在时间线打开请求 ${id}`" @click="openRequest(id, 'timeline')">时间线</button>
                      <button type="button" class="text-button" :aria-label="`在 Span 分析打开请求 ${id}`" @click="openRequest(id, 'span-profile')">Span 分析</button>
                      <button type="button" class="text-button" :aria-label="`在关键路径打开请求 ${id}`" @click="openRequest(id, 'critical-path')">关键路径</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section class="distribution-outliers">
            <header><div><strong>高延迟请求</strong><span>按延迟值降序</span></div><b>{{ sortedOutliers.length }}</b></header>
            <div class="distribution-table-wrap">
              <table class="distribution-table outlier-table">
                <thead><tr><th>排名</th><th>请求</th><th>延迟</th><th>百分位 / 尾部分类</th><th>打开请求视图</th></tr></thead>
                <tbody>
                  <tr v-if="!sortedOutliers.length"><td colspan="5" class="empty-cell">没有高延迟请求。</td></tr>
                  <tr v-for="outlier in sortedOutliers" v-else :key="`${outlier.rank}:${outlier.request_id}`">
                    <td class="numeric">#{{ outlier.rank }}</td>
                    <td><code :title="outlier.request_id">{{ shortId(outlier.request_id, 18) }}</code></td>
                    <td class="numeric"><strong>{{ formatDuration(outlier.value) }}</strong></td>
                    <td><span class="tail-classification" :class="{ tail: outlier.p95_tail }">P{{ outlier.percentile.toFixed(2) }} · {{ outlier.p95_tail ? "P95 尾部" : "非 P95 尾部" }}</span></td>
                    <td class="request-actions">
                      <button type="button" class="text-button" :aria-label="`在时间线打开请求 ${outlier.request_id}`" @click="openRequest(outlier.request_id, 'timeline')">时间线</button>
                      <button type="button" class="text-button" :aria-label="`在 Span 分析打开请求 ${outlier.request_id}`" @click="openRequest(outlier.request_id, 'span-profile')">Span 分析</button>
                      <button type="button" class="text-button" :aria-label="`在关键路径打开请求 ${outlier.request_id}`" @click="openRequest(outlier.request_id, 'critical-path')">关键路径</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    </AsyncState>
  </section>
</template>
