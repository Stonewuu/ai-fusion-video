import type { ExecutionRecordRef, ExecutionRecordKind } from "./api/execution-records";
import { usePipelineStore } from "./store/pipeline-store";

const recordKinds: ExecutionRecordKind[] = ["GENERATION", "TEXT", "CANVAS_MEDIA", "COMFY_TEST", "EPISODE_AUDIO", "AGENT_MODEL", "MCP"];

export function openExecutionRecord(target: ExecutionRecordRef) {
  const store = usePipelineStore.getState();
  store.setExpandedTaskId(`execution-record:${target.kind}:${encodeURIComponent(target.id)}`);
  store.setPanelExpanded(true);
}

export function selectedExecutionRecord(value: string | null): ExecutionRecordRef | null {
  if (!value?.startsWith("execution-record:")) {
    return null;
  }
  const [, kind, encoded] = value.split(":");
  if (!recordKinds.includes(kind as ExecutionRecordKind) || !encoded) {
    return null;
  }
  try {
    return { kind: kind as ExecutionRecordKind, id: decodeURIComponent(encoded) };
  } catch {
    return null;
  }
}
