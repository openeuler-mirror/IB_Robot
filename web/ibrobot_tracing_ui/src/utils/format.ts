const STAGE_LABELS: Record<string, string> = {
  obs_frame_ms: "观测采样",
  dispatch_to_infer_ms: "分发至推理",
  preprocess_ms: "预处理",
  inference_ms: "模型调用",
  postprocess_ms: "后处理",
  action_chunk_publish_ms: "动作块发布",
  dispatch_decode_ms: "分发解码",
  queue_refill_ms: "分发至队列补充",
  refill_to_execute_ms: "队列补充至执行",
  execute_publish_ms: "执行发布",
  cloud_roundtrip_ms: "云端往返",
  total_ms: "分发至首次执行",
};

export function stageLabel(value: string): string {
  return STAGE_LABELS[value] ?? value;
}

export function metricLabel(value: string): string {
  return ({ minimum: "min", maximum: "max", mean: "mean", p50: "p50", p95: "p95", p99: "p99" })[value] ?? value;
}

export function formatDuration(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (Math.abs(value) < 0.001) return `${(value * 1000).toFixed(1)} μs`;
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(digits)} ms`;
}

export function formatInteger(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : new Intl.NumberFormat("zh-CN").format(value);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const units = ["字节", "千字节", "兆字节", "吉字节"];
  let amount = value;
  let index = 0;
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024;
    index += 1;
  }
  return `${amount.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatTimestamp(value: string | number | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

export function shortId(value: string, length = 12): string {
  return value.length <= length ? value : `${value.slice(0, length)}…`;
}

export function eventComponent(event: { component_id?: string; fields: Record<string, unknown>; origin: { provider: string } }): string {
  return event.component_id || String(event.fields.component_id ?? "") || event.origin.provider || "—";
}

export function recordToData(value: object): Record<string, unknown> {
  return { ...value } as Record<string, unknown>;
}
