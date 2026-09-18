<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { storeToRefs } from "pinia";
import { useProfilerStore } from "../../stores/profiler";
import { formatDuration, recordToData } from "../../utils/format";
import AppIcon from "../AppIcon.vue";
import AsyncState from "../AsyncState.vue";

const store = useProfilerStore();
const { callTree, requestId, tabState, tabError, searchQuery } = storeToRefs(store);
const collapsed = ref(new Set<string>());

const rows = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase("zh-CN");
  const nodes = callTree.value?.nodes ?? [];
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const allowed = new Set(nodes.filter((node) => !query || `${node.name} ${node.component_id} ${node.origin} ${node.status}`.toLocaleLowerCase("zh-CN").includes(query)).map((node) => node.id));
  if (query) {
    for (const id of [...allowed]) {
      let parent = byId.get(id)?.parent_id;
      while (parent) {
        allowed.add(parent);
        parent = byId.get(parent)?.parent_id;
      }
    }
  }
  const result: { node: NonNullable<typeof callTree.value>["nodes"][number]; depth: number; hasChildren: boolean }[] = [];
  const visit = (ids: string[], depth: number) => {
    ids.forEach((id) => {
      const node = byId.get(id);
      if (!node || !allowed.has(id)) return;
      const children = node.child_ids.filter((childId) => allowed.has(childId));
      result.push({ node, depth, hasChildren: children.length > 0 });
      if (!collapsed.value.has(id)) visit(children, depth + 1);
    });
  };
  visit(callTree.value?.root_ids ?? [], 0);
  return result;
});

watch(requestId, () => (collapsed.value = new Set()));

function toggle(id: string): void {
  const next = new Set(collapsed.value);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  collapsed.value = next;
}

function inspect(index: number): void {
  const node = rows.value[index].node;
  store.selection = { kind: "span", title: node.name, subtitle: node.component_id, data: recordToData(node) };
}
</script>

<template>
  <AsyncState
    :state="tabState['span-profile']"
    :error="tabError['span-profile']"
    :empty="!requestId || !callTree?.nodes.length"
    :empty-title="requestId ? '该请求没有完整 Span' : '调用树需要请求范围'"
    :empty-message="requestId ? '该追踪可能只有旧版事件，或 Span 尚未闭合。' : '先选择一个 request_id 以构建父子关系。'"
    @retry="store.fetchTab('span-profile', true)"
  >
    <div class="call-tree-view">
      <div class="tree-header"><span>调用</span><span>组件</span><span>来源</span><span>状态</span><span>耗时</span></div>
      <div class="tree-scroll">
        <div v-if="!rows.length" class="empty-cell">过滤条件未命中 Span</div>
        <button v-for="(row, index) in rows" :key="row.node.id" class="tree-row" type="button" @click="inspect(index)">
          <span class="tree-name" :style="{ paddingLeft: `${8 + row.depth * 18}px` }">
            <i v-if="row.hasChildren" :class="{ collapsed: collapsed.has(row.node.id) }" @click.stop="toggle(row.node.id)"><AppIcon name="chevron" :size="13" /></i>
            <i v-else class="tree-leaf" />
            <strong>{{ row.node.name }}</strong>
          </span>
          <code :title="row.node.component_id">{{ row.node.component_id || "—" }}</code>
          <span>{{ row.node.origin === "user" ? "用户" : "内置" }}</span>
          <span :class="`status-${row.node.status}`">{{ row.node.cycle ? "父链成环" : row.node.orphan ? "父项缺失" : row.node.status === "ok" ? "正常" : row.node.status === "incomplete" ? "未完成" : row.node.status }}</span>
          <b class="numeric" :title="row.node.self_duration_ns === null ? '' : `自身 ${formatDuration(row.node.self_duration_ns / 1_000_000, 3)}`">{{ formatDuration(row.node.duration_ns === null ? null : row.node.duration_ns / 1_000_000, 3) }}</b>
        </button>
      </div>
    </div>
  </AsyncState>
</template>
