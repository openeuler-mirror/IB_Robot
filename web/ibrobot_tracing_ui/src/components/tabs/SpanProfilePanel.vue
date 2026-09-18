<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import type {
  AggregateSpanProfileNode,
  RequestSpanProfileNode,
  SpanAnalysisMode,
} from "../../api/types";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, formatInteger, recordToData } from "../../utils/format";
import {
  aggregateBreadcrumb,
  componentColor,
  layoutAggregateProfile,
  layoutRequestProfile,
  numericNs,
  profileNodeKeyAction,
  profileNodeMatches,
} from "../../utils/spanProfile";
import AppIcon from "../AppIcon.vue";
import AsyncState from "../AsyncState.vue";
import CallTreePanel from "./CallTreePanel.vue";

const store = useProfilerStore();
const {
  activeTab,
  requestId,
  searchQuery,
  spanAnalysisMode,
  spanProfile,
  tabError,
  tabState,
} = storeToRefs(store);

const RULER_HEIGHT = 28;
const PLOT_LEFT = 72;
const PLOT_RIGHT = 10;
const zoom = ref(1);
const focusId = ref("");
const chartScroller = ref<HTMLElement | null>(null);
const viewportWidth = ref(760);
const dragging = ref(false);
const hoverNode = ref<RequestSpanProfileNode | AggregateSpanProfileNode | null>(null);
const hoverPoint = ref({ x: 0, y: 0 });
const hoverLane = ref(0);
let resizeObserver: ResizeObserver | null = null;
let panStartX = 0;
let panStartScroll = 0;

const requestProfile = computed(() => spanProfile.value?.mode === "request" ? spanProfile.value : null);
const aggregateProfile = computed(() => spanProfile.value?.mode === "aggregate" ? spanProfile.value : null);
const chartWidth = computed(() => Math.round(Math.max(460, viewportWidth.value) * zoom.value));
const plotWidth = computed(() => Math.max(1, chartWidth.value - PLOT_LEFT - PLOT_RIGHT));

const requestLayout = computed(() => layoutRequestProfile(
  requestProfile.value?.nodes ?? [],
  requestProfile.value?.duration_ns ?? 0,
  plotWidth.value,
));
const aggregateLayout = computed(() => layoutAggregateProfile(
  aggregateProfile.value?.nodes ?? [],
  aggregateProfile.value?.roots ?? [],
  plotWidth.value,
  focusId.value,
));
const activeLayout = computed(() => spanAnalysisMode.value === "request" ? requestLayout.value : aggregateLayout.value);
const plotHeight = computed(() => Math.max(150, activeLayout.value.contentHeight));
const svgHeight = computed(() => RULER_HEIGHT + plotHeight.value + 8);
const requestRects = computed(() => requestLayout.value.rects);
const aggregateRects = computed(() => aggregateLayout.value.rects);
const breadcrumbs = computed(() => aggregateBreadcrumb(aggregateProfile.value?.nodes ?? [], focusId.value));

const rulerTicks = computed(() => Array.from({ length: 6 }, (_, index) => {
  const ratio = index / 5;
  const label = requestProfile.value
    ? formatNs(numericNs(requestProfile.value.duration_ns) * ratio)
    : `${Math.round(ratio * 100)}%`;
  return { ratio, label };
}));

const diagnosticItems = computed(() => {
  if (!spanProfile.value) return [];
  const counts = { ...spanProfile.value.totals.diagnostic_counts };
  spanProfile.value.diagnostics.forEach((diagnostic) => {
    if (!(diagnostic.code in counts)) counts[diagnostic.code] = diagnostic.occurrence_ids.length;
  });
  return Object.entries(counts)
    .filter(([, count]) => count > 0)
    .sort(([leftCode, leftCount], [rightCode, rightCount]) => rightCount - leftCount || leftCode.localeCompare(rightCode))
    .map(([code, count]) => ({
      code,
      count,
      title: spanProfile.value?.diagnostics.find((diagnostic) => diagnostic.code === code)?.message ?? code,
    }));
});

const hoverInfo = computed(() => {
  const node = hoverNode.value;
  if (!node) return null;
  if ("occurrence_id" in node) {
    return {
      title: node.name,
      subtitle: node.component_id || "未标记组件",
      lines: [
        `总墙钟 ${formatNs(numericNs(node.duration_ns))}`,
        `自身/未覆盖 ${formatNs(numericNs(node.uncovered_wall_ns))}`,
        `偏移 ${formatNs(numericNs(node.start_offset_ns))} · 深度 ${node.depth} · 全局通道 ${hoverLane.value}`,
        `状态 ${node.status}${node.diagnostics.length ? ` · ${node.diagnostics.join(", ")}` : ""}`,
      ],
    };
  }
  return {
    title: node.name,
    subtitle: node.component_id || "未标记组件",
    lines: [
      `占比 ${node.percentage.toFixed(2)}% · 包含 ${formatNs(numericNs(node.value_ns))}`,
      `自身/未覆盖 ${formatNs(numericNs(node.self_value_ns))}`,
      `${formatInteger(node.occurrence_count)} 次 · ${formatInteger(node.request_count)} 个请求`,
      `错误 ${formatInteger(node.error_count)} · 未完成 ${formatInteger(node.incomplete_count)} · 无效 ${formatInteger(node.invalid_count)}`,
    ],
  };
});

function formatNs(value: number): string {
  return formatDuration(value / 1_000_000, 3);
}

function bandY(y: number): number {
  return y;
}

function matches(node: RequestSpanProfileNode | AggregateSpanProfileNode): boolean {
  return profileNodeMatches(node, searchQuery.value);
}

function requestBorder(node: RequestSpanProfileNode): "danger" | "diagnostic" | "normal" {
  if (node.status === "error") return "danger";
  if (node.status === "incomplete" || node.diagnostics.length) return "diagnostic";
  return "normal";
}

function aggregateBorder(node: AggregateSpanProfileNode): "danger" | "diagnostic" | "normal" {
  if (node.error_count > 0) return "danger";
  if (node.incomplete_count > 0 || node.invalid_count > 0) return "diagnostic";
  return "normal";
}

function inspectRequest(node: RequestSpanProfileNode): void {
  store.selection = { kind: "span", title: node.name, subtitle: node.component_id, data: recordToData(node) };
}

function inspectAggregate(node: AggregateSpanProfileNode): void {
  store.selection = { kind: "span", title: node.name, subtitle: `${node.component_id} · 聚合路径`, data: recordToData(node) };
}

function requestNodeLabel(node: RequestSpanProfileNode): string {
  return `${node.name}，组件 ${node.component_id || "未标记"}，墙钟 ${formatNs(numericNs(node.duration_ns))}，状态 ${node.status}。按 Enter 或空格检查`;
}

function aggregateNodeLabel(node: AggregateSpanProfileNode): string {
  return `${node.name}，组件 ${node.component_id || "未标记"}，包含权重 ${formatNs(numericNs(node.value_ns))}，占比 ${node.percentage.toFixed(2)}%。按 Enter 或空格检查，按 Shift+Enter 聚焦子树`;
}

function handleRequestNodeKey(event: KeyboardEvent, node: RequestSpanProfileNode): void {
  if (profileNodeKeyAction(event.key, event.shiftKey, false) !== "inspect") return;
  event.preventDefault();
  inspectRequest(node);
}

function handleAggregateNodeKey(event: KeyboardEvent, node: AggregateSpanProfileNode): void {
  const action = profileNodeKeyAction(event.key, event.shiftKey, true);
  if (!action) return;
  event.preventDefault();
  if (action === "focus") void focusAggregate(node);
  else inspectAggregate(node);
}

function showTooltip(event: PointerEvent, node: RequestSpanProfileNode | AggregateSpanProfileNode, lane = 0): void {
  const scroller = chartScroller.value;
  const bounds = scroller?.getBoundingClientRect();
  hoverNode.value = node;
  hoverLane.value = lane;
  hoverPoint.value = {
    x: (scroller?.scrollLeft ?? 0) + Math.min((bounds?.width ?? 300) - 274, Math.max(8, event.clientX - (bounds?.left ?? 0) + 12)),
    y: (scroller?.scrollTop ?? 0) + Math.max(8, event.clientY - (bounds?.top ?? 0) + 12),
  };
}

function hideTooltip(): void {
  hoverNode.value = null;
}

async function setMode(mode: SpanAnalysisMode): Promise<void> {
  focusId.value = "";
  await store.setSpanAnalysisMode(mode);
}

async function setZoom(value: number): Promise<void> {
  const scroller = chartScroller.value;
  const oldCenterRatio = scroller && scroller.scrollWidth
    ? (scroller.scrollLeft + scroller.clientWidth / 2) / scroller.scrollWidth
    : 0.5;
  zoom.value = Math.min(8, Math.max(0.75, value));
  await nextTick();
  if (scroller) scroller.scrollLeft = oldCenterRatio * scroller.scrollWidth - scroller.clientWidth / 2;
}

async function fitChart(): Promise<void> {
  zoom.value = 1;
  await nextTick();
  chartScroller.value?.scrollTo({ left: 0 });
}

async function resetChart(): Promise<void> {
  focusId.value = "";
  await fitChart();
}

async function focusAggregate(node: AggregateSpanProfileNode): Promise<void> {
  focusId.value = node.id;
  await fitChart();
}

async function navigateFocus(id: string): Promise<void> {
  focusId.value = id;
  await fitChart();
}

function beginPan(event: PointerEvent): void {
  const target = event.target as Element;
  if (event.button !== 0 || target.closest(".profile-node")) return;
  const scroller = chartScroller.value;
  if (!scroller) return;
  dragging.value = true;
  panStartX = event.clientX;
  panStartScroll = scroller.scrollLeft;
  scroller.setPointerCapture(event.pointerId);
}

function movePan(event: PointerEvent): void {
  if (!dragging.value || !chartScroller.value) return;
  chartScroller.value.scrollLeft = panStartScroll - (event.clientX - panStartX);
}

function endPan(event: PointerEvent): void {
  if (!dragging.value) return;
  dragging.value = false;
  chartScroller.value?.releasePointerCapture(event.pointerId);
}

function handleWheel(event: WheelEvent): void {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  void setZoom(zoom.value * (event.deltaY > 0 ? 0.85 : 1.18));
}

function handleKeyboard(event: KeyboardEvent): void {
  if (activeTab.value !== "span-profile" || spanAnalysisMode.value === "call-tree") return;
  const target = event.target as HTMLElement;
  if (target.matches("input, select, textarea, [contenteditable='true']")) return;
  if (event.key === "+" || event.key === "=") {
    event.preventDefault();
    void setZoom(zoom.value * 1.25);
  } else if (event.key === "-") {
    event.preventDefault();
    void setZoom(zoom.value / 1.25);
  } else if (event.key === "0") {
    event.preventDefault();
    void resetChart();
  } else if (event.key.toLocaleLowerCase("en-US") === "f") {
    event.preventDefault();
    void fitChart();
  }
}

watch(spanProfile, (profile) => {
  if (profile?.mode !== "aggregate" || !profile.nodes.some((node) => node.id === focusId.value)) focusId.value = "";
});

watch(chartScroller, (current, previous) => {
  if (previous) resizeObserver?.unobserve(previous);
  if (current && resizeObserver) {
    viewportWidth.value = Math.max(320, current.clientWidth);
    resizeObserver.observe(current);
  }
});

onMounted(() => {
  resizeObserver = new ResizeObserver(([entry]) => {
    viewportWidth.value = Math.max(320, entry.contentRect.width);
  });
  if (chartScroller.value) resizeObserver.observe(chartScroller.value);
  window.addEventListener("keydown", handleKeyboard);
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  window.removeEventListener("keydown", handleKeyboard);
});
</script>

<template>
  <section class="span-profile-panel">
    <header class="profile-semantic-banner">
      <strong>Span 分析</strong>
      <span>{{ spanAnalysisMode === "call-tree" ? "显式父子关系与自身耗时。" : "工具化 Span 墙钟时间：包括等待、I/O 和异步暂停；不是 CPU 使用率。" }}</span>
    </header>

    <div class="profile-toolbar">
      <div class="profile-segment" aria-label="Span 分析模式">
        <button type="button" :class="{ active: spanAnalysisMode === 'call-tree' }" @click="setMode('call-tree')">调用树</button>
        <button type="button" :class="{ active: spanAnalysisMode === 'request' }" @click="setMode('request')">请求区间</button>
        <button type="button" :class="{ active: spanAnalysisMode === 'aggregate' }" @click="setMode('aggregate')">聚合冰柱</button>
      </div>
      <span class="profile-toolbar-spacer" />
      <span v-if="spanAnalysisMode !== 'call-tree'" class="profile-legend"><i class="inclusive" />包含权重 <i class="self" />自身/未覆盖 <i class="incomplete" />未完成</span>
      <div v-if="spanAnalysisMode !== 'call-tree'" class="profile-zoom" aria-label="剖析图缩放">
        <button type="button" title="缩小（-）" :disabled="zoom <= 0.75" @click="setZoom(zoom / 1.25)">−</button>
        <code>{{ Math.round(zoom * 100) }}%</code>
        <button type="button" title="放大（+）" :disabled="zoom >= 8" @click="setZoom(zoom * 1.25)">+</button>
        <button type="button" title="适合宽度（F）" @click="fitChart"><AppIcon name="fit" :size="13" />适宽</button>
        <button type="button" title="重置视图与聚合焦点（0）" @click="resetChart">重置</button>
      </div>
    </div>

    <div v-if="spanAnalysisMode === 'call-tree'" class="span-analysis-content"><CallTreePanel /></div>

    <template v-else>
    <div v-if="spanProfile" class="profile-stat-strip">
      <dl><dt>返回节点</dt><dd>{{ formatInteger(spanProfile.returned_nodes) }} / {{ formatInteger(spanProfile.total_nodes) }}</dd></dl>
      <dl><dt>Span 次数</dt><dd>{{ formatInteger(spanProfile.totals.occurrence_count) }}</dd></dl>
      <dl><dt>请求数</dt><dd>{{ formatInteger(spanProfile.totals.request_count) }}</dd></dl>
      <dl><dt>错误 / 未完成</dt><dd>{{ formatInteger(spanProfile.totals.error_count) }} / {{ formatInteger(spanProfile.totals.incomplete_count) }}</dd></dl>
      <dl><dt>选中墙钟并集</dt><dd>{{ formatNs(numericNs(spanProfile.totals.selected_wall_union_ns)) }}</dd></dl>
      <dl><dt>并发系数</dt><dd>{{ spanProfile.totals.concurrency_factor.toFixed(2) }}×</dd></dl>
    </div>

    <div v-if="spanProfile?.truncated" class="profile-truncation" role="status">
      <AppIcon name="warning" :size="14" />
      <strong>结果已截断</strong>
      <span>服务端因 {{ spanProfile.truncation_reason || "节点上限" }} 仅返回 {{ formatInteger(spanProfile.returned_nodes) }} / {{ formatInteger(spanProfile.total_nodes) }} 个节点，子树可能不完整。</span>
    </div>
    <div v-if="diagnosticItems.length" class="profile-diagnostics" aria-label="诊断摘要">
      <strong>诊断</strong>
      <span v-for="item in diagnosticItems" :key="item.code" :title="item.title"><code>{{ item.code }}</code> {{ formatInteger(item.count) }}</span>
    </div>
    <nav v-if="aggregateProfile && focusId" class="profile-breadcrumb" aria-label="聚合焦点路径">
      <button type="button" @click="navigateFocus('')">全部根</button>
      <template v-for="item in breadcrumbs" :key="item.id">
        <i>/</i><button type="button" :class="{ current: item.id === focusId }" @click="navigateFocus(item.id)">{{ item.name }}</button>
      </template>
    </nav>

    <div v-if="spanAnalysisMode === 'request' && !requestId" class="profile-prompt state-view">
      <AppIcon name="trace" :size="24" />
      <strong>请选择一个请求以生成 Span 剖析</strong>
      <span>单请求模式只绘制明确的 request_id；当前不会向服务端发出无效请求。可从“请求”标签或上方范围栏选择。</span>
      <button class="button secondary" type="button" @click="store.activeTab = 'requests'">打开请求列表</button>
    </div>
    <AsyncState
      v-else
      class="profile-async-state"
      :state="tabState['span-profile']"
      :error="tabError['span-profile']"
      :empty="Boolean(spanProfile && !spanProfile.nodes.length)"
      :empty-title="spanAnalysisMode === 'request' ? '该请求没有可剖析 Span' : '当前范围没有可聚合 Span'"
      empty-message="调整请求、组件或搜索范围后重试。"
      loading-message="正在计算 Span 墙钟剖析…"
      @retry="store.fetchTab('span-profile', true)"
    >
      <div
        ref="chartScroller"
        class="profile-chart-scroll"
        :class="{ dragging }"
        @pointerdown="beginPan"
        @pointermove="movePan"
        @pointerup="endPan"
        @pointercancel="endPan"
        @wheel="handleWheel"
      >
        <svg
          v-if="spanProfile"
          class="profile-chart"
          :width="chartWidth"
          :height="svgHeight"
          :viewBox="`0 0 ${chartWidth} ${svgHeight}`"
          role="img"
          :aria-label="spanAnalysisMode === 'request' ? '按时间排列的单请求 Span 墙钟冰柱图' : '按包含墙钟权重分区的聚合 Span 冰柱图'"
        >
          <defs>
            <pattern id="profile-incomplete-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="rgba(255,255,255,.75)" stroke-width="2" />
            </pattern>
          </defs>
          <rect class="profile-chart-background" :width="chartWidth" :height="svgHeight" />
          <g class="profile-ruler">
            <line :x1="PLOT_LEFT" :x2="chartWidth - PLOT_RIGHT" :y1="RULER_HEIGHT - 7" :y2="RULER_HEIGHT - 7" />
            <g v-for="tick in rulerTicks" :key="tick.ratio" :transform="`translate(${PLOT_LEFT + tick.ratio * plotWidth}, 0)`">
              <line y1="15" :y2="svgHeight" />
              <text :text-anchor="tick.ratio === 0 ? 'start' : tick.ratio === 1 ? 'end' : 'middle'" y="11">{{ tick.label }}</text>
            </g>
          </g>
          <g class="profile-depth-bands">
            <g v-for="band in activeLayout.depthBands" :key="band.depth">
              <rect
                :x="PLOT_LEFT"
                 :y="RULER_HEIGHT + bandY(band.y)"
                :width="plotWidth"
                :height="band.height"
                :class="{ alternate: band.depth % 2 }"
              />
               <text x="6" :y="RULER_HEIGHT + bandY(band.y) + 14">D{{ band.depth }}<tspan v-if="band.laneCount > 1" x="6" dy="11">{{ band.laneCount }} lanes</tspan></text>
            </g>
          </g>

          <g v-if="requestProfile">
            <g
              v-for="rect in requestRects"
              :key="rect.node.id"
              class="profile-node"
              :class="[
                `border-${requestBorder(rect.node)}`,
                { 'search-match': searchQuery && matches(rect.node), 'search-dim': searchQuery && !matches(rect.node) },
              ]"
              role="button"
              tabindex="0"
              :aria-label="requestNodeLabel(rect.node)"
              @click.stop="inspectRequest(rect.node)"
              @keydown="handleRequestNodeKey($event, rect.node)"
              @pointerenter="showTooltip($event, rect.node, rect.lane)"
              @pointermove="showTooltip($event, rect.node, rect.lane)"
              @pointerleave="hideTooltip"
            >
              <rect
                :x="PLOT_LEFT + rect.x"
                :y="RULER_HEIGHT + rect.y"
                :width="rect.width"
                :height="rect.height"
                :fill="componentColor(rect.node.component_id)"
              />
              <rect
                v-if="rect.node.status === 'incomplete' || rect.node.duration_ns === null"
                class="profile-hatch"
                :x="PLOT_LEFT + rect.x"
                :y="RULER_HEIGHT + rect.y"
                :width="rect.width"
                :height="rect.height"
              />
              <rect
                class="profile-self-weight"
                :x="PLOT_LEFT + rect.x"
                :y="RULER_HEIGHT + rect.y + rect.height - 3"
                :width="rect.selfWidth"
                height="3"
              />
              <svg
                v-if="rect.width > 16"
                class="profile-clipped-label"
                :x="PLOT_LEFT + rect.x + 4"
                :y="RULER_HEIGHT + rect.y"
                :width="Math.max(0, rect.width - 8)"
                :height="rect.height - 3"
                overflow="hidden"
              ><text x="0" y="15">{{ rect.node.name }}</text></svg>
            </g>
          </g>

          <g v-else-if="aggregateProfile">
            <g
              v-for="rect in aggregateRects"
              :key="rect.node.id"
              class="profile-node"
              :class="[
                `border-${aggregateBorder(rect.node)}`,
                { 'search-match': searchQuery && matches(rect.node), 'search-dim': searchQuery && !matches(rect.node) },
              ]"
              role="button"
              tabindex="0"
              :aria-label="aggregateNodeLabel(rect.node)"
              @click.stop="inspectAggregate(rect.node)"
              @dblclick.stop="focusAggregate(rect.node)"
              @keydown="handleAggregateNodeKey($event, rect.node)"
              @pointerenter="showTooltip($event, rect.node)"
              @pointermove="showTooltip($event, rect.node)"
              @pointerleave="hideTooltip"
            >
              <rect
                :x="PLOT_LEFT + rect.x"
                :y="RULER_HEIGHT + rect.y"
                :width="rect.width"
                :height="rect.height"
                :fill="componentColor(rect.node.component_id)"
              />
              <rect
                v-if="rect.node.incomplete_count || rect.node.invalid_count"
                class="profile-hatch"
                :x="PLOT_LEFT + rect.x"
                :y="RULER_HEIGHT + rect.y"
                :width="rect.width"
                :height="rect.height"
              />
              <rect
                class="profile-self-weight"
                :x="PLOT_LEFT + rect.x"
                :y="RULER_HEIGHT + rect.y + rect.height - 3"
                :width="rect.selfWidth"
                height="3"
              />
              <svg
                v-if="rect.width > 16"
                class="profile-clipped-label"
                :x="PLOT_LEFT + rect.x + 4"
                :y="RULER_HEIGHT + rect.y"
                :width="Math.max(0, rect.width - 8)"
                :height="rect.height - 3"
                overflow="hidden"
              ><text x="0" y="15">{{ rect.node.name }} · {{ rect.node.percentage.toFixed(1) }}%</text></svg>
            </g>
          </g>
        </svg>
        <aside
          v-if="hoverInfo"
          class="profile-tooltip"
          :style="{ left: `${hoverPoint.x}px`, top: `${hoverPoint.y}px` }"
        >
          <strong>{{ hoverInfo.title }}</strong><code>{{ hoverInfo.subtitle }}</code>
          <span v-for="line in hoverInfo.lines" :key="line">{{ line }}</span>
          <small>{{ aggregateProfile ? "双击或 Shift+Enter 聚焦子树 · 单击或 Enter 检查" : "单击、Enter 或空格在检查器中查看" }}</small>
        </aside>
      </div>
    </AsyncState>
    </template>
  </section>
</template>
