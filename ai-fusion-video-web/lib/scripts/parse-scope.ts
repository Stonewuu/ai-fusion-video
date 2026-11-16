import type { ScriptEpisode } from "@/lib/api/script";

export type ScriptParseScope = "pending" | "range" | "all";

export class ScriptParseRangeError extends Error {
  constructor(
    public field: "from" | "to" | "range",
    message: string,
  ) {
    super(message);
  }
}

export function selectParseEpisodes(
  episodes: ScriptEpisode[],
  scope: ScriptParseScope,
  from: number,
  to: number,
  replaceExisting: boolean,
) {
  if (scope === "range") {
    if (!Number.isSafeInteger(from) || from < 1) {
      throw new ScriptParseRangeError("from", "开始集数须为大于 0 的整数");
    }
    if (!Number.isSafeInteger(to) || to < 1) {
      throw new ScriptParseRangeError("to", "结束集数须为大于 0 的整数");
    }
    if (to < from) {
      throw new ScriptParseRangeError("range", "开始集数不能大于结束集数");
    }
  }
  return episodes.filter((episode) => {
    if (!episode.rawContentLength) {
      return false;
    }
    if (
      scope === "range" &&
      (episode.episodeNumber < from || episode.episodeNumber > to)
    ) {
      return false;
    }
    return replaceExisting || episode.parsingStatus !== 2;
  });
}

export function parseEpisodesMessage(
  episodes: ScriptEpisode[],
  replaceExisting: boolean,
  scope: ScriptParseScope,
) {
  const ids = episodes.map((episode) => episode.id).join(", ");
  const range =
    scope === "range"
      ? `请完成以下 ${episodes.length} 集的结构化解析，scriptEpisodeIds 为 [${ids}]。`
      : scope === "all"
        ? "请重新解析当前剧本的全部有原文分集。"
        : "请完成当前剧本全部有原文且尚未解析完成的分集。";
  return (
    range +
    "分页读取分集目录，按对应数据库 ID 逐集读取原文，保存场次、对白、资产关联和本集梗概；每批完成后继续下一批，直到本次全部分集处理完成。" +
    (replaceExisting
      ? "用户已确认重新解析这些分集。每集先读取原文并准备首批场次，首批保存时替换该集旧场次，后续追加，不重复清空。保留分集 ID。"
      : "只解析尚未完成的分集；启动后已完成的分集自动跳过，未完成分集从已有进度继续，不重复覆盖或追加已保存场次。") +
    "不修改本次范围外的分集，不把全量原文放入主任务上下文。"
  );
}
