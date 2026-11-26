import type { ScriptSourcePreview, ScriptSourceSegment, ScriptSplitRule } from "@/lib/api/script";
import { MAX_EPISODE_LENGTH, MAX_SOURCE_EPISODES, MAX_SOURCE_LENGTH } from "./source-import-limits";

export interface SourceSplitRequest {
  source: string;
  rule: ScriptSplitRule;
  targetLength: number;
}

export type SourceSplitResponse = { preview: ScriptSourcePreview } | { error: string };

enum Rank {
  Special,
  Episode,
  Chapter,
  Section,
  Volume,
  Numbered,
}

interface Heading {
  start: number;
  end: number;
  title: string;
  rank: Rank;
  volumeStart: number;
}

const number = "[0-9零〇一二三四五六七八九十百千万廿卅卌两兩壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]+";
// 标题沿用 Java 拆分器的空白和换行规则。粘贴进来的 Unicode 分隔符也会影响章节边界。
const titleChar = "[^\\r\\n\\u0085\\u2028\\u2029]";
const titleSpaces = /[\t-\r \u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+/g;
const whitespace = "[\\t-\\r\\u001c-\\u0020\\u1680\\u2000-\\u2006\\u2008-\\u200a\\u2028\\u2029\\u205f\\u3000]";
const blank = new RegExp(`^${whitespace}*$`);
const outerWhitespace = new RegExp(`^${whitespace}+|${whitespace}+$`, "g");
const chinese = new RegExp(`^第?${number}([集话話章回节節幕卷部篇])(?:[：:、.．—-]|$|(?=[^0-9]))${titleChar}*$`, "u");
const english = /^(chapter|chap\.?|episode|ep\.?|part|book|volume)(?:[\t-\r ]+|(?=[0-9]))(?:[0-9]+|[ivxlcdm]+)\b[^\r\n\u0085\u2028\u2029]*$/i;
const numbered = new RegExp(`^${number}[、.．:：][\\t-\\r ]*[^\\t-\\r ]${titleChar}{0,70}$`, "u");
const special = new RegExp(`^(序章|序言|前言|楔子|引子|序幕|尾声|尾聲|终章|終章|后记|後記|番外)(?:[\\t-\\r ：:、—-]${titleChar}{0,70})?$`, "u");
const contentsEntry = /(?:\.{2,}|…{2,}|·{3,})[\t-\r ]*\d+[\t-\r ]*$/;

function headingRank(title: string): Rank | undefined {
  const plain = title.replace(/[【】\[\]（）()]/g, "").replace(outerWhitespace, "");
  const unit = chinese.exec(plain.replace(titleSpaces, ""))?.[1];
  if (unit && !contentsEntry.test(title)) {
    if ("集话話".includes(unit)) {
      return Rank.Episode;
    }
    if ("章回".includes(unit)) {
      return Rank.Chapter;
    }
    return "节節幕".includes(unit) ? Rank.Section : Rank.Volume;
  }
  const prefix = english.exec(plain)?.[1].toLowerCase();
  if (prefix) {
    if (prefix.startsWith("ep")) {
      return Rank.Episode;
    }
    return prefix.startsWith("chap") ? Rank.Chapter : Rank.Volume;
  }
  if (special.test(plain)) {
    return Rank.Special;
  }
  return numbered.test(plain) ? Rank.Numbered : undefined;
}

function selectHeadings(source: string, rule: ScriptSplitRule): Heading[] {
  if (rule === "LENGTH") {
    return [];
  }
  const headings: Heading[] = [];
  let volume = "";
  let volumeStart = -1;
  let bestRank: Rank | undefined;
  for (const line of source.matchAll(/[^\r\n]+/g)) {
    if (line[0].length > 160) {
      continue;
    }
    const title = line[0].normalize("NFKC").replaceAll("\uFEFF", "")
      .replace(outerWhitespace, "").replace(/^#{1,6}[\t-\r ]*/, "").replace(outerWhitespace, "");
    const rank = headingRank(title);
    if (rank === undefined) {
      continue;
    }
    if (rank === Rank.Volume) {
      volume = title;
      volumeStart = line.index;
    }
    const withVolume = rank < Rank.Volume;
    headings.push({
      start: line.index,
      end: line.index + line[0].length,
      title: withVolume && volume ? `${volume} · ${title}` : title,
      rank,
      volumeStart: withVolume ? volumeStart : -1,
    });
    if (withVolume) {
      volumeStart = -1;
    }
    if (rank > Rank.Special && (bestRank === undefined || rank < bestRank)) {
      bestRank = rank;
    }
  }

  const rank = rule === "CHAPTER" ? Rank.Chapter : rule === "EPISODE" ? Rank.Episode : bestRank ?? Rank.Special;
  const candidates = headings.filter((item) => item.rank === rank || item.rank === Rank.Special);
  if (rule !== "AUTO" && !candidates.some((item) => item.rank === rank)) {
    throw new Error("未识别到所选标题，请选择自动识别或按字数分集");
  }

  const laterTitles = new Set<string>();
  const selected: Heading[] = [];
  let nextStart = source.length;
  for (let i = candidates.length - 1; i >= 0; i--) {
    const heading = candidates[i];
    const start = rank < Rank.Volume && heading.volumeStart >= 0 ? heading.volumeStart : heading.start;
    const key = heading.title.replace(titleSpaces, "").toLowerCase();
    // 只有后文另有同名标题且此处没有正文时，才把这行当作目录项。
    if (!laterTitles.has(key) || !blank.test(source.slice(heading.end, nextStart))) {
      selected.push({ ...heading, start });
    }
    laterTitles.add(key);
    nextStart = start;
  }
  return selected.reverse();
}

function splitBoundary(source: string, start: number, target: number): number {
  const limit = start + target;
  const middle = start + Math.floor(target / 2);
  let cut = 0;
  for (let i = limit - 1; i >= middle; i--) {
    if (source[i] === "\n") {
      cut = i + 1;
      break;
    }
    if (!cut && i > middle && "。！？.!?\r".includes(source[i])) {
      cut = i + 1;
    }
  }
  if (!cut) {
    cut = limit;
  }
  if (/[\uD800-\uDBFF]/.test(source[cut - 1]) || (source[cut - 1] === "\r" && source[cut] === "\n")) {
    cut--;
  }
  return cut;
}

function appendSegments(segments: ScriptSourceSegment[], source: string, start: number, end: number, title: string, target: number) {
  const divided = end - start > target;
  let part = 1;
  while (start < end) {
    if (segments.length >= MAX_SOURCE_EPISODES) {
      throw new Error("分集超过 20000 集，请调整分集方式或每集字数");
    }
    const cut = start + target < end ? splitBoundary(source, start, target) : end;
    const partTitle = divided ? `${title}（${part}）` : title;
    if (partTitle.length > 256) {
      throw new Error("分集标题超过 256 字，请缩短原文中的标题后重试");
    }
    segments.push({ start, end: cut, title: partTitle });
    start = cut;
    part++;
  }
}

function splitSource({ source, rule, targetLength }: SourceSplitRequest): ScriptSourcePreview {
  if (source.length > MAX_SOURCE_LENGTH || blank.test(source)) {
    throw new Error("请提供 2000 万字以内的原文");
  }
  if (!Number.isInteger(targetLength) || targetLength < 1000 || targetLength > MAX_EPISODE_LENGTH) {
    throw new Error("每集字数应在 1000 至 100000 之间");
  }
  const headings = selectHeadings(source, rule);
  const segments: ScriptSourceSegment[] = [];
  let start = 0;
  let title = "第 1 集";
  if (headings.length > 0) {
    const first = headings[0];
    if (!blank.test(source.slice(0, first.start))) {
      appendSegments(segments, source, 0, first.start, "前言", targetLength);
      start = first.start;
    }
    title = first.title;
    for (const next of headings.slice(1)) {
      appendSegments(segments, source, start, next.start, title, targetLength);
      start = next.start;
      title = next.title;
    }
  }
  appendSegments(segments, source, start, source.length, title, targetLength);
  return {
    segments,
    totalChars: source.length,
    headingCount: headings.length,
    splitCount: Math.max(0, segments.length - Math.max(1, headings.length)),
  };
}

self.onmessage = ({ data }: MessageEvent<SourceSplitRequest>) => {
  try {
    self.postMessage({ preview: splitSource(data) } satisfies SourceSplitResponse);
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : "分集识别失败，请重试" } satisfies SourceSplitResponse);
  }
};
