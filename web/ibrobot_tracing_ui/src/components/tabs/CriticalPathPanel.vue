<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import type { CriticalPathBottleneck, CriticalPathSegment, JsSafeInteger } from "../../api/types";
import { useProfilerStore } from "../../stores/profiler";
import { buildCriticalPathGeometry } from "../../utils/criticalPath";
import { formatDuration, formatInteger, recordToData, shortId } from "../../utils/format";
import { componentColor, numericNs } from "../../utils/spanProfile";
import AppIcon from "../AppIcon.vue";
import AsyncState from "../AsyncState.vue";

const store = useProfilerStore();
const { criticalPath, requestId, tabError, tabState } = storeToRefs(store);
const selectedSegmentId = ref("");

const PLOT_LEFT = 68;
const PLOT_RIGHT = 14;
const SVG_WIDTH = 1_000;
const TRACK_Y = 34;
const TRACK_HEIGHT = 34;
const plotWidth = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT;

const geometry = computed(() => buildCriticalPathGeometry(
  criticalPath.value?.segments ?? [],
  criticalPath.value?.duration_ns ?? 0,
  criticalPath.value?.returned_duration_ns ?? 0,
  criticalPath.value?.omitted_duration_ns ?? 0,
  plotWidth,
));
const orderedSegments = computed(() => geometry.value.rects.map((rect) => rect.segment));
const bottlenecks = computed(() => [...(criticalPath.value?.bottlenecks ?? [])].sort((left, right) => left.rank - right.rank));
const diagnostics = computed(() => [...(criticalPath.value?.diagnostics ?? [])].sort((left, right) => right.count - left.count || left.code.localeCompare(right.code)));
const axisTicks = computed(() => Array.from({ length: 5 }, (_, index) => {
  const ratio = index / 4;
  return {
    ratio,
    label: formatNs(numericNs(criticalPath.value?.duration_ns) * ratio),
  };
}));

function formatNs(value: JsSafeInteger | null | undefined): string {
  return formatDuration(numericNs(value) / 1_000_000, 3);
}

function kindLabel(kind: CriticalPathSegment["kind"] | CriticalPathBottleneck["kind"]): string {
  if (kind === "span") return "Span";
  if (kind === "flow") return "Flow";
  return "未归因间隙";
}

function owner(segment: CriticalPathSegment | CriticalPathBottleneck): string {
  return segment.component_id || segment.edge_id || "—";
}

function kindTotal(kind: CriticalPathSegment["kind"]): string {
  return formatNs(criticalPath.value?.totals.by_kind_ns[kind] ?? 0);
}

function segmentTitle(segment: CriticalPathSegment): string {
  return `${kindLabel(segment.kind)} · ${segment.label || "未归因"} · ${formatNs(segment.duration_ns)} · ${segment.percentage.toFixed(2)}% · ${owner(segment)}`;
}

function inspect(segment: CriticalPathSegment): void {
  selectedSegmentId.value = segment.id;
  store.selection = {
    kind: segment.kind === "unattributed" ? "critical-path" : segment.kind,
    title: segment.label || "未归因墙钟间隙",
    subtitle: `${kindLabel(segment.kind)} · ${formatNs(segment.duration_ns)} · ${owner(segment)}`,
    data: recordToData(segment),
  };
}

function segmentFill(segment: CriticalPathSegment): string {
  return segment.kind === "span" ? componentColor(segment.component_id) : `url(#critical-path-${segment.kind})`;
}

watch(criticalPath, () => {
  selectedSegmentId.value = "";
});
</script>

<template>
  <section class="critical-path-panel">
    <header class="critical-path-semantic-banner">
      <div><strong>Request Critical Wall Path</strong><b>Instrumented Wall Time</b></div>
      <p>墙钟归因包含等待、I/O 与异步暂停；它不是 CPU 时间，也不是跨主机调度 DAG。</p>
    </header>

    <div v-if="!requestId" class="critical-path-prompt state-view">
      <AppIcon name="trace" :size="24" />
      <strong>请选择一个请求以生成关键墙钟路径</strong>
      <span>关键路径只接受明确的 request_id；当前不会向服务端发出无效请求。可从“请求”标签、延迟分布或上方范围栏选择。</span>
      <button class="button secondary" type="button" @click="store.activeTab = 'requests'">打开请求列表</button>
    </div>

    <AsyncState
      v-else
      class="critical-path-async"
      :state="tabState['critical-path']"
      :error="tabError['critical-path']"
      :empty="tabState['critical-path'] === 'ready' && !criticalPath"
      empty-title="没有关键路径结果"
      empty-message="该请求没有可用的墙钟边界或归因记录。"
      loading-message="正在计算请求关键墙钟路径…"
      @retry="store.fetchTab('critical-path', true)"
    >
      <div v-if="criticalPath" class="critical-path-scroll">
        <div class="critical-path-stat-strip" aria-label="关键路径摘要">
          <dl><dt>请求</dt><dd :title="criticalPath.request_id">{{ shortId(criticalPath.request_id, 18) }}</dd></dl>
          <dl class="boundary-stat"><dt>边界来源</dt><dd :title="criticalPath.boundary_source">{{ criticalPath.boundary_source || "—" }}</dd></dl>
          <dl><dt>总墙钟</dt><dd>{{ formatNs(criticalPath.duration_ns) }}</dd></dl>
          <dl><dt>归因覆盖</dt><dd>{{ criticalPath.totals.coverage_percent.toFixed(2) }}%</dd></dl>
          <dl><dt>Span</dt><dd>{{ kindTotal("span") }}</dd></dl>
          <dl><dt>Flow</dt><dd>{{ kindTotal("flow") }}</dd></dl>
          <dl><dt>Gap</dt><dd>{{ kindTotal("unattributed") }}</dd></dl>
          <dl><dt>返回分段</dt><dd>{{ formatInteger(criticalPath.returned_segments) }} / {{ formatInteger(criticalPath.total_segments) }}</dd></dl>
        </div>

        <div v-if="criticalPath.truncated" class="critical-path-truncation" role="status">
          <AppIcon name="warning" :size="14" />
          <strong>仅显示按时间排序的返回前缀</strong>
          <span>已返回 {{ formatInteger(criticalPath.returned_segments) }} 个分段（{{ formatNs(criticalPath.returned_duration_ns) }}）；尾部省略 {{ formatInteger(criticalPath.omitted_segments) }} 个分段，精确时长 {{ formatNs(criticalPath.omitted_duration_ns) }}。原因：<code>{{ criticalPath.truncation_reason }}</code></span>
        </div>

        <div v-if="diagnostics.length" class="critical-path-diagnostic-strip" aria-label="关键路径诊断摘要">
          <strong>诊断</strong>
          <span v-for="item in diagnostics" :key="item.code" :title="item.message"><code>{{ item.code }}</code> {{ formatInteger(item.count) }}</span>
        </div>

        <section class="critical-path-chart-section" aria-labelledby="critical-path-chart-heading">
          <header>
            <div>
              <strong id="critical-path-chart-heading">墙钟分区时间线</strong>
              <span>{{ criticalPath.boundary_start_source || "未知起点" }} → {{ criticalPath.boundary_end_source || "未知终点" }}</span>
            </div>
            <span class="critical-path-legend"><i class="span" />Span 组件 <i class="flow" />Flow <i class="gap" />Gap <i v-if="criticalPath.truncated" class="omitted" />省略尾部</span>
          </header>
          <div v-if="numericNs(criticalPath.duration_ns) > 0" class="critical-path-chart-wrap">
            <svg
              class="critical-path-chart"
              :viewBox="`0 0 ${SVG_WIDTH} 94`"
              role="img"
              aria-labelledby="critical-path-chart-title critical-path-chart-description"
            >
              <title id="critical-path-chart-title">请求关键墙钟路径</title>
              <desc id="critical-path-chart-description">按请求完整边界比例显示不重叠的 Span、Flow 和未归因墙钟分段；截断时单独标出未返回的尾部。</desc>
              <defs>
                <pattern id="critical-path-flow" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="8" height="8" fill="#d9f2f3" /><line x1="0" y1="0" x2="0" y2="8" stroke="#008b96" stroke-width="4" />
                </pattern>
                <pattern id="critical-path-unattributed" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="8" height="8" fill="#eceef1" /><line x1="0" y1="0" x2="0" y2="8" stroke="#9aa0a8" stroke-width="3" />
                </pattern>
                <pattern id="critical-path-omitted" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="7" height="7" fill="#fff4e8" /><line x1="0" y1="0" x2="0" y2="7" stroke="#e66a00" stroke-width="2" />
                </pattern>
              </defs>
              <g class="critical-path-axis">
                <line :x1="PLOT_LEFT" :x2="SVG_WIDTH - PLOT_RIGHT" y1="74" y2="74" />
                <g v-for="tick in axisTicks" :key="tick.ratio" :transform="`translate(${PLOT_LEFT + tick.ratio * plotWidth}, 0)`">
                  <line y1="25" y2="78" />
                  <text :text-anchor="tick.ratio === 0 ? 'start' : tick.ratio === 1 ? 'end' : 'middle'" y="89">{{ tick.label }}</text>
                </g>
              </g>
              <rect class="critical-path-track" :x="PLOT_LEFT" :y="TRACK_Y" :width="plotWidth" :height="TRACK_HEIGHT" />
              <g
                v-for="rect in geometry.rects"
                :key="rect.segment.id"
                class="critical-path-segment"
                :class="{ selected: selectedSegmentId === rect.segment.id }"
                role="button"
                tabindex="0"
                :aria-label="`${segmentTitle(rect.segment)}。按回车或空格检查。`"
                @click="inspect(rect.segment)"
                @keydown.enter.prevent="inspect(rect.segment)"
                @keydown.space.prevent="inspect(rect.segment)"
              >
                <title>{{ segmentTitle(rect.segment) }}</title>
                <rect
                  :x="PLOT_LEFT + rect.x"
                  :y="TRACK_Y"
                  :width="rect.width"
                  :height="TRACK_HEIGHT"
                  :fill="segmentFill(rect.segment)"
                />
                <svg
                  v-if="rect.width > 54"
                  class="critical-path-clipped-label"
                  :x="PLOT_LEFT + rect.x + 4"
                  :y="TRACK_Y"
                  :width="Math.max(0, rect.width - 8)"
                  :height="TRACK_HEIGHT"
                  overflow="hidden"
                ><text x="0" y="21">{{ rect.segment.label || "Gap" }}</text></svg>
              </g>
              <g v-if="criticalPath.truncated && geometry.omitted" class="critical-path-omitted-tail">
                <title>省略尾部：{{ formatInteger(criticalPath.omitted_segments) }} 个分段，{{ formatNs(criticalPath.omitted_duration_ns) }}</title>
                <rect :x="PLOT_LEFT + geometry.omitted.x" :y="TRACK_Y" :width="geometry.omitted.width" :height="TRACK_HEIGHT" />
                <text v-if="geometry.omitted.width > 84" :x="PLOT_LEFT + geometry.omitted.x + 6" :y="TRACK_Y + 21">OMITTED · {{ formatNs(criticalPath.omitted_duration_ns) }}</text>
              </g>
            </svg>
          </div>
          <div v-else class="critical-path-zero" role="status">请求边界时长为 0；没有可按比例绘制的墙钟区间。</div>
        </section>

        <div class="critical-path-detail-grid">
          <section class="critical-path-bottlenecks">
            <header><div><strong>排序瓶颈</strong><span>按归因墙钟总量</span></div><b>{{ formatInteger(bottlenecks.length) }}</b></header>
            <div class="critical-path-table-wrap">
              <table>
                <thead><tr><th>排名</th><th>记录</th><th>位置</th><th>时长</th><th>占比</th><th>分段</th></tr></thead>
                <tbody>
                  <tr v-if="!bottlenecks.length"><td colspan="6" class="empty-cell">没有可排序的已归因瓶颈。</td></tr>
                  <tr v-for="item in bottlenecks" v-else :key="`${item.rank}:${item.kind}:${item.source_id}`">
                    <td class="numeric">#{{ item.rank }}</td>
                    <td><strong :title="item.label">{{ item.label || "—" }}</strong><small>{{ kindLabel(item.kind) }} · {{ item.status || "—" }}</small></td>
                    <td><code :title="owner(item)">{{ owner(item) }}</code></td>
                    <td class="numeric">{{ formatNs(item.duration_ns) }}</td>
                    <td class="numeric">{{ item.percentage.toFixed(2) }}%</td>
                    <td class="numeric">{{ formatInteger(item.segment_count) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section class="critical-path-segments-section">
            <header><div><strong>按时间分段</strong><span>单击行在安全检查器中查看字段与诊断</span></div><b>{{ formatInteger(orderedSegments.length) }}</b></header>
            <div class="critical-path-table-wrap segment-table-wrap">
              <table class="critical-path-segment-table">
                <thead><tr><th>#</th><th>偏移</th><th>记录</th><th>时长</th><th>占比</th><th>类型</th><th>组件 / 边</th><th>状态</th></tr></thead>
                <tbody>
                  <tr v-if="!orderedSegments.length"><td colspan="8" class="empty-cell">边界内没有可返回的墙钟分段；请检查诊断。</td></tr>
                  <tr
                    v-for="segment in orderedSegments"
                    v-else
                    :key="segment.id"
                    role="button"
                    tabindex="0"
                    :aria-label="`${segmentTitle(segment)}。打开检查器。`"
                    :class="{ selected: selectedSegmentId === segment.id }"
                    @click="inspect(segment)"
                    @keydown.enter.prevent="inspect(segment)"
                    @keydown.space.prevent="inspect(segment)"
                  >
                    <td class="numeric">{{ segment.index + 1 }}</td>
                    <td class="numeric">{{ formatNs(segment.offset_ns) }}</td>
                    <td><strong :title="segment.label">{{ segment.label || "未归因" }}</strong><small v-if="segment.diagnostics.length" :title="segment.diagnostics.join(', ')">{{ segment.diagnostics.join(", ") }}</small></td>
                    <td class="numeric">{{ formatNs(segment.duration_ns) }}</td>
                    <td class="numeric">{{ segment.percentage.toFixed(2) }}%</td>
                    <td><span class="critical-path-kind" :class="segment.kind">{{ kindLabel(segment.kind) }}</span></td>
                    <td><code :title="owner(segment)">{{ owner(segment) }}</code></td>
                    <td>{{ segment.status || "—" }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section v-if="diagnostics.length" class="critical-path-diagnostics">
            <header><div><strong>诊断详情</strong><span>边界、记录排除与时钟完整性</span></div><b>{{ formatInteger(diagnostics.length) }}</b></header>
            <ul>
              <li v-for="item in diagnostics" :key="item.code">
                <code>{{ item.code }}</code><strong>{{ formatInteger(item.count) }}</strong><span>{{ item.message }}</span><small v-if="item.source_ids.length" :title="item.source_ids.join(', ')">{{ item.source_ids.map((id) => shortId(id, 16)).join(", ") }}</small>
              </li>
            </ul>
          </section>
        </div>
      </div>
    </AsyncState>
  </section>
</template>
