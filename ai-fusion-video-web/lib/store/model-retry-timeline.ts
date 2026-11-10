import type { AiChatStreamEvent } from "@/lib/api/ai-assistant";
import type { SubTimelineItem, TimelineItem } from "./pipeline-timeline";

/** 重试只丢弃尚未完成的模型轮次；已完成工具和其他子任务保持原样。 */
export function applyModelRetry(timeline: TimelineItem[], event: AiChatStreamEvent): TimelineItem[] {
  if (!event.content) {
    throw new Error("模型重试事件缺少提示内容");
  }
  const keep = (item: SubTimelineItem) => {
    if (!event.replyId || item.type === "retry") {
      return true;
    }
    return item.type === "tool"
      ? item.batchId !== event.replyId || !["preparing", "calling"].includes(item.status)
      : item.replyId !== event.replyId;
  };
  const notice: SubTimelineItem = { type: "retry", text: event.content };
  if (event.parentToolCallId) {
    return timeline.map((item) => item.type === "tool" && item.id === event.parentToolCallId
      ? { ...item, children: [...(item.children ?? []).filter(keep), notice] }
      : item);
  }
  return [...timeline.filter(keep), notice];
}
