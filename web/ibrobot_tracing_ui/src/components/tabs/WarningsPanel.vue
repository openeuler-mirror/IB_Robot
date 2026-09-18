<script setup lang="ts">
import { computed } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import AsyncState from "../AsyncState.vue";
import AppIcon from "../AppIcon.vue";

const store = useProfilerStore();
const { warnings, warningTotal, warningsTruncated, tabState, tabError, searchQuery } = storeToRefs(store);
const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  return warnings.value.filter((warning) => !query || warning.toLocaleLowerCase("zh-CN").includes(query));
});
function inspect(warning: string, index: number): void {
  store.selection = { kind: "warning", title: `警告 ${index + 1}`, subtitle: "分析器诊断", data: { message: warning } };
}
</script>

<template>
  <AsyncState :state="tabState.warnings" :error="tabError.warnings" :empty="!warnings.length" empty-title="没有分析警告" empty-message="Span 与 Flow 边界完整，分析器未发现结构问题。" @retry="store.fetchTab('warnings', true)">
    <div class="warnings-view">
      <p v-if="warningsTruncated" role="status">共 {{ warningTotal }} 条诊断，仅展示前 {{ warnings.length }} 条；筛选仅作用于已展示记录。</p>
      <header><strong>分析器警告</strong><span>这些记录不会阻止查看已有数据，但可能影响统计完整性。</span></header>
      <button v-for="(warning, index) in rows" :key="`${index}:${warning}`" type="button" @click="inspect(warning, index)"><AppIcon name="warning" /><span>{{ warning }}</span><b>{{ index + 1 }}</b></button>
      <div v-if="!rows.length" class="empty-cell">过滤条件未命中警告</div>
    </div>
  </AsyncState>
</template>
