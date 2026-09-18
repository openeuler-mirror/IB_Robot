import type { InspectorSelection, TracepointDefinition } from "../api/types";

function value(data: Record<string, unknown>, key: string): string {
  const item = data[key];
  return item === null || item === undefined ? "" : String(item);
}

function fieldOrigin(data: Record<string, unknown>): string {
  const fields = data.fields;
  if (!fields || typeof fields !== "object" || Array.isArray(fields)) return "built-in";
  return String((fields as Record<string, unknown>).origin || "built-in");
}

export function resolveTracepointDescription(
  selection: InspectorSelection | null,
  definitions: TracepointDefinition[],
): string {
  if (!selection) return "";
  if (selection.kind === "component") return value(selection.data, "description");

  const timelineItem = "label" in selection.data;
  const kind = timelineItem ? value(selection.data, "kind") : selection.kind;
  if (kind !== "event" && kind !== "span") return "";
  const name = value(selection.data, timelineItem ? "label" : "name");
  const componentId = value(selection.data, "component_id");
  const origin = timelineItem || kind === "event"
    ? fieldOrigin(selection.data)
    : value(selection.data, "origin");

  return definitions.find((definition) =>
    definition.kind === kind
    && definition.component_id === componentId
    && definition.name === name
    && definition.origin === origin
  )?.description ?? "";
}
