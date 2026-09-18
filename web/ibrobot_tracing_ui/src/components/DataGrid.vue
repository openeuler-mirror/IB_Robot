<script setup lang="ts">
import { computed, ref, watch } from "vue";

export interface GridColumn {
  key: string;
  label: string;
  width?: string;
  align?: "left" | "right" | "center";
  className?: string;
  format?: (value: unknown, row: Record<string, unknown>) => string;
}

const props = withDefaults(
  defineProps<{
    rows: Record<string, unknown>[];
    columns: GridColumn[];
    rowKey: string;
    selectedKey?: string;
    emptyText?: string;
  }>(),
  { selectedKey: "", emptyText: "没有匹配记录" },
);

const emit = defineEmits<{ select: [row: Record<string, unknown>] }>();
const cursor = ref(-1);
const visibleRows = computed(() => props.rows.slice(0, 2000));

watch(
  () => props.selectedKey,
  (key) => {
    if (key) cursor.value = visibleRows.value.findIndex((row) => String(row[props.rowKey]) === key);
  },
  { immediate: true },
);

function move(delta: number): void {
  if (!visibleRows.value.length) return;
  cursor.value = Math.max(0, Math.min(visibleRows.value.length - 1, cursor.value + delta));
  document.getElementById(`grid-row-${String(visibleRows.value[cursor.value][props.rowKey])}`)?.scrollIntoView({ block: "nearest" });
}

function activate(): void {
  if (cursor.value >= 0 && visibleRows.value[cursor.value]) emit("select", visibleRows.value[cursor.value]);
}
</script>

<template>
  <div
    class="data-grid-wrap"
    tabindex="0"
    @keydown.j.prevent="move(1)"
    @keydown.k.prevent="move(-1)"
    @keydown.down.prevent="move(1)"
    @keydown.up.prevent="move(-1)"
    @keydown.enter.prevent="activate"
  >
    <table class="data-grid">
      <colgroup><col v-for="column in columns" :key="column.key" :style="{ width: column.width }" /></colgroup>
      <thead><tr><th v-for="column in columns" :key="column.key" :class="`align-${column.align || 'left'}`">{{ column.label }}</th></tr></thead>
      <tbody>
        <tr v-if="!visibleRows.length"><td :colspan="columns.length" class="empty-cell">{{ emptyText }}</td></tr>
        <tr
          v-for="(row, index) in visibleRows"
          v-else
          :id="`grid-row-${String(row[rowKey])}`"
          :key="String(row[rowKey])"
          :class="{ selected: String(row[rowKey]) === selectedKey, cursor: index === cursor }"
          @click="cursor = index; emit('select', row)"
        >
          <td
            v-for="column in columns"
            :key="column.key"
            :class="[`align-${column.align || 'left'}`, column.className]"
            :title="column.format ? column.format(row[column.key], row) : String(row[column.key] ?? '')"
          >
            <slot :name="`cell-${column.key}`" :row="row" :value="row[column.key]">
              {{ column.format ? column.format(row[column.key], row) : (row[column.key] ?? "—") }}
            </slot>
          </td>
        </tr>
      </tbody>
    </table>
    <div v-if="rows.length > visibleRows.length" class="grid-limit">为保证交互流畅，仅显示前 {{ visibleRows.length }} 条，共 {{ rows.length }} 条。</div>
  </div>
</template>
