import type { AiChatStreamEvent } from "@/lib/api/ai-assistant";
import type { TimelineItem } from "./pipeline-timeline";

/** 完整消息按回复身份替换已显示的片段，工具已经开始时仍保留原先的位置。 */
export function replaceMessageSnapshot(
  timeline: TimelineItem[],
  event: AiChatStreamEvent,
): TimelineItem[] {
  if (event.parentToolCallId) {
    return timeline.map((item) => {
      if (item.type !== "tool" || item.id !== event.parentToolCallId) {
        return item;
      }
      return { ...item, children: replaceReply(item.children ?? [], event) };
    });
  }
  return replaceReply(timeline, event);
}

function replaceReply(items: TimelineItem[], event: AiChatStreamEvent): TimelineItem[] {
  const updated: TimelineItem[] = [];
  let contentSeen = false;
  let reasoningSeen = false;
  for (const item of items) {
    if (item.type === "content" && item.replyId === event.replyId) {
      if (!contentSeen && event.content) {
        updated.push({ ...item, text: event.content });
      }
      contentSeen = true;
    } else if (item.type === "reasoning" && item.replyId === event.replyId) {
      if (!reasoningSeen && event.reasoningContent) {
        updated.push({
          ...item,
          text: event.reasoningContent,
          durationMs: event.reasoningDurationMs ?? item.durationMs,
        });
      }
      reasoningSeen = true;
    } else {
      updated.push(item);
    }
  }
  const missing: TimelineItem[] = [];
  if (!reasoningSeen && event.reasoningContent) {
    missing.push({ type: "reasoning", text: event.reasoningContent, replyId: event.replyId,
      durationMs: event.reasoningDurationMs ?? undefined });
  }
  if (!contentSeen && event.content) {
    missing.push({ type: "content", text: event.content, replyId: event.replyId });
  }
  const toolIndex = updated.findIndex((item) => item.type === "tool" && item.batchId === event.replyId);
  updated.splice(toolIndex < 0 ? updated.length : toolIndex, 0, ...missing);
  return updated;
}
