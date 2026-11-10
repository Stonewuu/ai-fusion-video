import type { ModelPreset } from "@/lib/api/ai-model";
import type { ModelsDevEntry } from "@/lib/api/models-dev";

const INPUT_LABELS: Record<string, string> = { text: "文本", image: "图片", video: "视频", audio: "音频", file: "文件", pdf: "PDF" };

export function ModelPresetDetails({ preset, online }: { preset?: ModelPreset; online?: ModelsDevEntry }) {
  const config = preset?.config;
  const context = preset?.contextWindow ?? online?.contextWindow;
  const output = preset?.outputTokenLimit ?? online?.outputTokenLimit;
  const reasoning = preset?.supportReasoning ?? online?.supportReasoning;
  const rows = [
    ["上下文", context == null ? "—" : context.toLocaleString()],
    ["最大输出", output == null ? "—" : output.toLocaleString()],
    ["思考", reasoning == null ? "—" : reasoning ? "支持" : "不支持"],
    ["输入", (preset?.multimodalInputTypes ?? online?.inputModalities)?.map(value => INPUT_LABELS[value] || value).join(" · ") || "—"],
    ...(preset?.modelType === 2 || online?.modelType === 2 ? [
      ["参考图", config?.maxReferenceImages == null ? "—" : `最多 ${config.maxReferenceImages} 张`],
      ["分辨率", config?.supportedSizes ? Object.keys(config.supportedSizes).join(" · ") : "—"],
    ] : []),
    ...(preset?.modelType === 3 || online?.modelType === 3 ? [
      ["分辨率", Array.isArray(config?.supportedResolutions) ? config.supportedResolutions.join(" · ") : "—"],
      ["时长", config?.maxDuration == null ? "—" : `最长 ${config.maxDuration} 秒`],
    ] : []),
  ];
  return (
    <div className="flex min-w-0 flex-col gap-4">
      <div>
        <h3 className="break-words font-medium">{online?.name ?? preset?.name}</h3>
        <p className="mt-1 break-all text-xs text-muted-foreground">{online?.modelId ?? preset?.modelCode ?? preset?.code}</p>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-3 text-sm">
        {rows.map(([label, value]) => <div key={label} className="contents"><dt className="text-muted-foreground">{label}</dt><dd className="min-w-0 break-words">{value}</dd></div>)}
      </dl>
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-ring">更多信息</summary>
        <div className="flex flex-col gap-2 p-1 pt-3">
          {preset?.description && <p>{preset.description}</p>}
          {Array.isArray(config?.supportedAspectRatios) && <p>比例：{config.supportedAspectRatios.join(" · ")}</p>}
          {online && <p>来源：Models.dev · {online.providerName}</p>}
          {preset?.sourceUrls?.[0] && <a href={preset.sourceUrls[0]} target="_blank" rel="noreferrer" className="underline">官方参数</a>}
          <p>更新日期：{online?.updatedAt ?? preset?.verifiedAt ?? "—"}</p>
        </div>
      </details>
    </div>
  );
}
