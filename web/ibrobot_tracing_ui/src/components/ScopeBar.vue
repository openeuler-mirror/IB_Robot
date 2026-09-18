<script setup lang="ts">
import { storeToRefs } from "pinia";
import { computed } from "vue";
import { useProfilerStore } from "../stores/profiler";
import { shortId } from "../utils/format";
import AppIcon from "./AppIcon.vue";

const store = useProfilerStore();
const { requests, components, requestId, componentId, metric, graphView, searchQuery, motionPaused } = storeToRefs(store);
const scopeComponents = computed(() => components.value.filter((component) => !["operation", "instant_event"].includes(component.kind)));

function clearScope(): void {
  requestId.value = "";
  componentId.value = "";
  searchQuery.value = "";
  store.selection = null;
}
</script>

<template>
  <section class="scope-bar" aria-label="分析范围">
    <span class="scope-title">分析范围</span>
    <label>
      <span>请求</span>
      <select v-model="requestId" aria-label="请求范围">
        <option value="">全部请求</option>
        <option v-for="request in requests" :key="request.request_id" :value="request.request_id">
          {{ shortId(request.request_id, 18) }}
        </option>
      </select>
    </label>
    <label>
      <span>组件</span>
      <select v-model="componentId" aria-label="组件范围">
        <option value="">全部组件</option>
        <option v-for="component in scopeComponents" :key="component.component_id" :value="component.component_id">
          {{ component.name }}
        </option>
      </select>
    </label>
    <label>
      <span>指标</span>
      <select v-model="metric" aria-label="拓扑聚合指标">
        <option value="minimum">min</option>
        <option value="p50">p50</option>
        <option value="p95">p95</option>
        <option value="p99">p99</option>
        <option value="maximum">max</option>
        <option value="mean">mean</option>
      </select>
    </label>
    <label>
      <span>拓扑</span>
      <select v-model="graphView" aria-label="拓扑层级">
        <option value="nodes">ROS 节点</option>
        <option value="components">组件</option>
        <option value="tracepoints">埋点</option>
      </select>
    </label>
    <label class="scope-search">
      <AppIcon name="search" :size="14" />
      <input id="global-trace-search" v-model="searchQuery" type="search" placeholder="过滤拓扑与当前表格" aria-label="过滤拓扑与当前表格" />
      <kbd>/</kbd>
    </label>
    <button class="icon-button" type="button" :title="motionPaused ? '恢复流向动画' : '暂停流向动画'" @click="motionPaused = !motionPaused">
      <AppIcon :name="motionPaused ? 'play' : 'pause'" />
      <span class="sr-only">{{ motionPaused ? "恢复流向动画" : "暂停流向动画" }}</span>
    </button>
    <button v-if="requestId || componentId || searchQuery" class="text-button" type="button" @click="clearScope">清除范围</button>
  </section>
</template>
