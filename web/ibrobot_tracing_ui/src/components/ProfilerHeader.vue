<script setup lang="ts">
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../stores/profiler";
import { formatBytes, formatTimestamp } from "../utils/format";
import AppIcon from "./AppIcon.vue";

const store = useProfilerStore();
const { sources, sourcesState, sourcesError, sourceId, selectedSource, analysisState, loadJob } = storeToRefs(store);

defineEmits<{ analyze: []; refresh: [] }>();
</script>

<template>
  <header class="profiler-header">
    <div class="brand" title="IB-Robot Tracing">
      <span class="brand-mark"><AppIcon name="trace" :size="18" /></span>
      <strong>IB-Robot Tracing</strong>
    </div>
    <div class="source-controls">
      <label class="sr-only" for="trace-source">追踪源</label>
      <select id="trace-source" v-model="sourceId" :disabled="sourcesState === 'loading' || !sources.length">
        <option v-if="!sources.length" value="">{{ sourcesState === "loading" ? "正在发现追踪源…" : "没有可用追踪源" }}</option>
        <option v-for="source in sources" :key="source.id" :value="source.id">{{ source.name }} · {{ source.kind.toUpperCase() }}</option>
      </select>
      <button class="icon-button" type="button" title="重新扫描追踪源" :disabled="sourcesState === 'loading'" @click="$emit('refresh')">
        <AppIcon name="refresh" />
        <span class="sr-only">重新扫描追踪源</span>
      </button>
      <button class="button primary" type="button" :disabled="!sourceId || analysisState === 'loading'" @click="$emit('analyze')">
        <AppIcon name="play" />
        载入分析
      </button>
    </div>
    <div class="header-context" :title="selectedSource?.name || sourcesError">
      <template v-if="loadJob && analysisState === 'loading'">
        <span>{{ loadJob.status === "queued" ? `排队等待${loadJob.queue_position ? `（第 ${loadJob.queue_position} 位）` : ""}` : "正在解析追踪" }}</span>
        <span class="job-progress indeterminate"><i /></span>
      </template>
      <template v-else-if="selectedSource">
        <span class="path-text">{{ selectedSource.name }} · {{ formatBytes(selectedSource.size_bytes) }} · {{ selectedSource.file_count }} 个文件</span>
        <span v-if="selectedSource.modified_at">{{ formatTimestamp(selectedSource.modified_at) }}</span>
      </template>
      <span v-else-if="sourcesError" class="danger-text">{{ sourcesError }}</span>
    </div>
  </header>
</template>
