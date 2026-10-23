"use client";

import { create } from "zustand";
import type { AiChatStreamEvent } from "@/lib/api/ai-assistant";
import type { CanvasResourceRef } from "./types";

export interface CanvasAssistantContext {
  canvasId: string;
  selectedNodeIds: string[];
  visibleNodeIds: string[];
  focusedResource: CanvasResourceRef | null;
  lastSequence: number;
}
export interface CanvasAssistantCapture {
  title: string;
  projectId: number | null;
  context: CanvasAssistantContext;
}
export interface CanvasAssistantBinding {
  canvasId: string;
  title: string;
  userId: number;
  projectId: number | null;
  selectedCount: number;
  prepare: () => Promise<CanvasAssistantCapture>;
  focus: (ids: string[]) => void;
}
interface Session {
  binding: CanvasAssistantBinding | null;
  following: boolean;
  highlight: { canvasId: string; nodeIds: string[]; key: string } | null;
  setFollowing: (following: boolean) => void;
}
export const useCanvasAssistantSession = create<Session>((set) => ({
  binding: null, following: true, highlight: null,
  setFollowing: (following) => set({ following }),
}));
export const EMPTY_AGENT_NODES: string[] = [];
const captures = new Map<string, CanvasAssistantCapture>();
const drafts = new Map<string, () => Promise<void>>();
const UI_TOOLS = new Set(["apply_canvas_operations", "edit_canvas_resource", "arrange_canvas_nodes", "bind_canvas_output",
  "undo_canvas_command", "focus_canvas_nodes", "create_canvas_checkpoint"]);
let highlightTimer: ReturnType<typeof setTimeout> | undefined;
let captureUserId: number | null = null;

export function attachCanvasAssistant(binding: CanvasAssistantBinding) {
  if (captureUserId !== binding.userId) { captures.clear(); captureUserId = binding.userId; }
  useCanvasAssistantSession.setState({ binding });
  return () => {
    if (useCanvasAssistantSession.getState().binding === binding) useCanvasAssistantSession.setState({ binding: null });
  };
}
export function registerCanvasAssistantDraft(key: string, save: () => Promise<void>) {
  drafts.set(key, save);
  return () => { if (drafts.get(key) === save) drafts.delete(key); };
}
export async function saveCanvasAssistantDrafts(keys: string[], errorMessage = "选中内容仍有未保存或冲突的草稿，请打开对应节点处理后再发送。") {
  for (const key of keys) {
    const save = drafts.get(key);
    if (save) await save();
    if (sessionStorage.getItem(key)) throw new Error(errorMessage);
  }
}
export function rememberCanvasAssistantCapture(conversationId: string, capture: CanvasAssistantCapture) {
  captures.set(conversationId, capture);
  if (captures.size > 100) captures.delete(captures.keys().next().value!);
}
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function nodeIds(value: unknown) {
  return Array.isArray(value) ? value.flatMap((node) => record(node) && typeof node.id === "string" ? [node.id] : []) : [];
}
export function receiveCanvasAssistantResult(conversationId: string, event: AiChatStreamEvent) {
  if (event.outputType !== "TOOL_FINISHED" || !event.toolResult || !event.toolName || !UI_TOOLS.has(event.toolName)) return;
  const capture = captures.get(conversationId);
  const session = useCanvasAssistantSession.getState();
  if (!capture || session.binding?.canvasId !== capture.context.canvasId) return;
  let result: unknown;
  try { result = JSON.parse(event.toolResult); } catch { return; }
  if (!record(result) || result.status !== "success") return;
  const receipt = record(result.receipt) ? result.receipt : null;
  const changes = receipt && record(receipt.changes) ? receipt.changes : null;
  const ids = [...new Set([...nodeIds(result.affectedNodes), ...nodeIds(changes?.upsertedNodes),
    ...(typeof result.targetNodeId === "string" ? [result.targetNodeId] : [])])];
  if (ids.length) {
    const key = `${event.runId}:${event.sequence}`;
    useCanvasAssistantSession.setState({ highlight: { canvasId: capture.context.canvasId, nodeIds: ids, key } });
    clearTimeout(highlightTimer);
    highlightTimer = setTimeout(() => {
      if (useCanvasAssistantSession.getState().highlight?.key === key) useCanvasAssistantSession.setState({ highlight: null });
    }, 4000);
  }
  if (session.following && record(result.focus) && result.focus.canvasId === capture.context.canvasId
    && result.focus.agentRunId === event.runId && Array.isArray(result.focus.nodeIds)
    && result.focus.nodeIds.every((id) => typeof id === "string")) {
    session.binding.focus(result.focus.nodeIds);
  }
}
