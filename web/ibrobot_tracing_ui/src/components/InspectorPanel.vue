<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../stores/profiler";
import AppIcon from "./AppIcon.vue";

const store = useProfilerStore();
const { selection, selectionDescription } = storeToRefs(store);

const entries = computed(() => (selection.value ? Object.entries(selection.value.data).filter(([key]) => key !== "description") : []));

function display(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}
</script>

<template>
  <aside v-if="selection" class="inspector" aria-label="记录检查器">
    <header>
      <div><span>检查器</span><strong>{{ selection.title }}</strong><small v-if="selection.subtitle">{{ selection.subtitle }}</small></div>
      <button class="icon-button" type="button" title="关闭检查器" @click="store.selection = null"><AppIcon name="close" /><span class="sr-only">关闭检查器</span></button>
    </header>
    <section v-if="selectionDescription" class="inspector-description">
      <strong>埋点说明</strong>
      <p>{{ selectionDescription }}</p>
    </section>
    <dl>
      <template v-for="([key, value]) in entries" :key="key">
        <dt>{{ key }}</dt>
        <dd :class="{ structured: typeof value === 'object' && value !== null }">{{ display(value) }}</dd>
      </template>
    </dl>
  </aside>
</template>
