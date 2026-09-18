<script setup lang="ts">
import type { LoadState } from "../api/types";
import AppIcon from "./AppIcon.vue";

withDefaults(
  defineProps<{
    state: LoadState;
    empty?: boolean;
    error?: string;
    emptyTitle?: string;
    emptyMessage?: string;
    loadingMessage?: string;
  }>(),
  {
    empty: false,
    error: "",
    emptyTitle: "没有匹配数据",
    emptyMessage: "调整当前范围或载入其他追踪源。",
    loadingMessage: "正在读取追踪数据…",
  },
);

defineEmits<{ retry: [] }>();
</script>

<template>
  <div v-if="state === 'loading'" class="state-view state-loading" aria-live="polite">
    <div class="loading-line" /><div class="loading-line short" /><div class="loading-line" />
    <span>{{ loadingMessage }}</span>
  </div>
  <div v-else-if="state === 'error'" class="state-view" role="alert">
    <AppIcon name="warning" :size="22" />
    <strong>数据读取失败</strong>
    <span>{{ error }}</span>
    <button class="button secondary" type="button" @click="$emit('retry')">重试</button>
  </div>
  <div v-else-if="empty || state === 'idle'" class="state-view">
    <AppIcon name="trace" :size="24" />
    <strong>{{ emptyTitle }}</strong>
    <span>{{ emptyMessage }}</span>
  </div>
  <slot v-else />
</template>
