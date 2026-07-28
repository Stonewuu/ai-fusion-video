import { resolveMediaUrl } from "@/lib/api/client";
import type { PipelineTask } from "@/lib/store/pipeline-store";

const TASK_MEDIA_URL_LINE_REGEXP = /((?:视频地址|下载地址)[:：]\s*)(https?:\/\/[^\s)]+|\/media\/[^\s)]+)/g;

export interface TaskMediaLinkInfo {
  label: string;
  rawUrl: string;
  resolvedUrl: string;
}

export function parseTaskContent(content: string): {
  markdownContent: string;
  mediaLinks: TaskMediaLinkInfo[];
} {
  const mediaLinks: TaskMediaLinkInfo[] = [];
  const markdownLines = content
    .split(/\r?\n/)
    .map((line) => {
      let extracted = false;
      const strippedLine = line.replace(
        TASK_MEDIA_URL_LINE_REGEXP,
        (_match, prefix, rawUrl) => {
          const resolvedUrl = resolveMediaUrl(rawUrl) || rawUrl;
          mediaLinks.push({
            label: prefix.replace(/[:：]\s*$/, "").trim() || "下载地址",
            rawUrl,
            resolvedUrl,
          });
          extracted = true;
          return "";
        }
      );

      if (!extracted) {
        return line;
      }

      return strippedLine.replace(/[·:：\s-]+$/g, "").trimEnd();
    });

  return {
    markdownContent: markdownLines
      .join("\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim(),
    mediaLinks,
  };
}

export function formatElapsed(durationMs: number): string {
  const secs = Math.floor(durationMs / 1000);
  if (secs < 60) return `${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m${secs % 60}s`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m`;
}

export function getElapsedStr(task: PipelineTask): string {
  const endTime = task.finishedAt ?? Date.now();
  return formatElapsed(endTime - task.createdAt);
}

export function formatDatetime(dateStr?: string): string {
  if (!dateStr) return "";
  try {
    return formatTimestamp(new Date(dateStr).getTime());
  } catch {
    return dateStr;
  }
}

export function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  const now = new Date();
  const time = d.toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
  });

  if (d.toDateString() === now.toDateString()) {
    return `今天 ${time}`;
  }

  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) {
    return `昨天 ${time}`;
  }

  if (d.getFullYear() === now.getFullYear()) {
    return `${d.getMonth() + 1}月${d.getDate()}日 ${time}`;
  }

  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 ${time}`;
}
