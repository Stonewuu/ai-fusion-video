import type { ScriptSourceSegment } from "@/lib/api/script";

export const MAX_SOURCE_LENGTH = 20_000_000;
export const MAX_SOURCE_FILE_BYTES = 100 * 1024 * 1024;
export const MAX_EPISODE_LENGTH = 100_000;

export function titleKey(title: string): string {
  return title.normalize("NFKC").replace(/\s+/g, "").toLocaleLowerCase();
}

export function duplicateTitles(titles: string[]): Set<string> {
  const seen = new Set<string>();
  const duplicates = new Set<string>();
  for (const title of titles) {
    const key = titleKey(title);
    if (key && seen.has(key)) {
      duplicates.add(key);
    }
    seen.add(key);
  }
  return duplicates;
}

export function splitSegment(
  segments: ScriptSourceSegment[],
  index: number,
  offset: number,
): ScriptSourceSegment[] {
  const item = segments[index];
  if (!item || offset <= 0 || offset >= item.end - item.start) {
    throw new Error("请把光标放在本集原文的分界位置");
  }
  const boundary = item.start + offset;
  return [
    ...segments.slice(0, index),
    { ...item, end: boundary },
    { start: boundary, end: item.end, title: `${item.title}（续）` },
    ...segments.slice(index + 1),
  ];
}

export function mergeSegment(
  segments: ScriptSourceSegment[],
  index: number,
): ScriptSourceSegment[] {
  const item = segments[index];
  const next = segments[index + 1];
  if (!item || !next) {
    throw new Error("已经是最后一集");
  }
  if (next.end - item.start > MAX_EPISODE_LENGTH) {
    throw new Error("合并后超过 10 万字，请调整分界");
  }
  return [
    ...segments.slice(0, index),
    { ...item, end: next.end },
    ...segments.slice(index + 2),
  ];
}

export function deleteSegment(
  source: string,
  segments: ScriptSourceSegment[],
  index: number,
): { source: string; segments: ScriptSourceSegment[] } {
  const removed = segments[index];
  const length = removed.end - removed.start;
  return {
    source: source.slice(0, removed.start) + source.slice(removed.end),
    segments: segments.flatMap((segment, current) => {
      if (current === index) {
        return [];
      }
      if (current > index) {
        return [
          {
            ...segment,
            start: segment.start - length,
            end: segment.end - length,
          },
        ];
      }
      return [segment];
    }),
  };
}

export async function readSourceFile(
  file: File,
  encoding: string,
): Promise<string> {
  if (!/\.(txt|md|markdown)$/i.test(file.name)) {
    throw new Error("请选择 TXT 或 Markdown 文件");
  }
  if (file.size > MAX_SOURCE_FILE_BYTES) {
    throw new Error("文件不能超过 100 MB");
  }
  let text: string;
  try {
    text = new TextDecoder(encoding, { fatal: true }).decode(
      await file.arrayBuffer(),
    );
  } catch {
    throw new Error("无法按所选编码读取文件，请切换文件编码后重试");
  }
  if (text.length > MAX_SOURCE_LENGTH) {
    throw new Error("原文不能超过 2000 万字");
  }
  if (!text.trim()) {
    throw new Error("文件没有可导入的文字");
  }
  if (text.includes("\u0000")) {
    throw new Error("文件含有无法识别的字符，请检查文件编码");
  }
  return text;
}
