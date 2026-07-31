"use client";

import { resolveMediaUrl } from "@/lib/api/client";
import { GenerationModelCapabilitiesResult } from "@/components/dashboard/generation-model-capabilities-result";
import {
  ImageGenerateResult,
  ThumbImage,
  VideoGenerateResult,
} from "@/components/dashboard/generation-media-result";
import { ToolMutationResult } from "@/components/dashboard/shared/tool-mutation-result";
import { assetTypeNames } from "../shared/ai-task-display";

// ========== 常量 ==========

/** 各工具字段中文标签 */
const toolResultLabels: Record<string, Record<string, string>> = {
  _common: {
    status: "状态",
    message: "消息",
    total: "总数",
    id: "ID",
    name: "名称",
    type: "类型",
    description: "描述",
    count: "数量",
    success: "成功",
    error: "错误",
    created: "已创建",
    updated: "已更新",
    imageUrl: "图片",
    thumbnailUrl: "缩略图",
    coverUrl: "封面图",
    prompt: "提示词",
    aiPrompt: "AI 提示词",
    width: "宽度",
    height: "高度",
    fileSize: "文件大小",
    sourceType: "来源",
    duration: "时长(秒)",
    videoUrl: "视频",
  },
  generate_image: {
    imageUrl: "生成图片",
    prompt: "使用提示词",
    status: "状态",
  },
  generate_video: {
    videoUrl: "生成视频",
    coverUrl: "封面图",
    prompt: "使用提示词",
    duration: "时长(秒)",
    status: "状态",
  },
  update_asset_image: {
    assetId: "资产 ID",
    assetName: "资产名称",
    itemId: "子资产 ID",
    itemType: "子资产类型",
    message: "结果",
  },
  add_asset_item: {
    assetItemId: "子资产 ID",
    assetId: "资产 ID",
    assetName: "资产名称",
    itemType: "子资产类型",
    message: "结果",
  },
  list_project_assets: {
    assets: "资产列表",
    itemCount: "子项数",
    coverUrl: "封面",
  },
  query_asset_metadata: {
    properties: "属性列表",
    propertyName: "属性名",
    propertyType: "属性类型",
    required: "必填",
  },
};

// ========== 工具函数 ==========

/** 将值友好化展示 */
function formatValue(val: unknown): string {
  if (val === null || val === undefined) return "—";
  if (typeof val === "boolean") return val ? "是" : "否";
  if (typeof val === "number") return String(val);
  if (typeof val === "string") return val.length > 120 ? val.slice(0, 120) + "…" : val;
  if (Array.isArray(val)) return `[${val.length} 项]`;
  if (typeof val === "object") return `{${Object.keys(val as object).length} 个字段}`;
  return String(val);
}

/** 获取字段中文标签 */
function getFieldLabel(toolName: string, key: string): string {
  return toolResultLabels[toolName]?.[key] ?? toolResultLabels._common[key] ?? key;
}

/** 判断是否为图片 URL */
function isImageUrl(val: unknown): boolean {
  if (typeof val !== "string" || !val) return false;
  if (/^(https?:\/\/|\/|data:image\/)/.test(val)) {
    return /\.(png|jpe?g|gif|webp|bmp|svg)(\?.*)?$/i.test(val)
      || val.startsWith("data:image/")
      || val.includes("/image/");
  }
  return false;
}

/** 状态值友好展示 */
function friendlyStatus(val: unknown): string {
  if (val === "success" || val === "ok") return "✅ 成功";
  if (val === "error") return "❌ 失败";
  return formatValue(val);
}

// ========== 子组件 ==========

type Obj = Record<string, unknown>;

/** 资产列表结果 — list_project_assets */
function AssetListResult({ data }: { data: unknown }) {
  const obj = data as Obj;
  const assets = (obj.assets as Array<Obj>) ?? [];
  const total = (obj.total as number) ?? assets.length;
  const typeStr = obj.type as string | undefined;
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">
        共 <span className="font-medium text-foreground">{total}</span> 项
        {typeStr && typeStr !== "all" && <span>（类型：{assetTypeNames[typeStr] ?? typeStr}）</span>}
      </p>
      {assets.length > 0 && (
        <ul className="space-y-1">
          {assets.slice(0, 10).map((asset, i) => (
            <li key={String(asset.id ?? i)} className="flex items-center gap-2 text-xs text-muted-foreground/90">
              <span className="w-1 h-1 rounded-full bg-muted-foreground/40 shrink-0" />
              <span className="font-medium text-foreground">{String(asset.name ?? "未命名")}</span>
              {!!asset.type && (
                <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                  {assetTypeNames[String(asset.type)] ?? String(asset.type)}
                </span>
              )}
              {asset.itemCount != null && <span className="text-[10px]">{String(asset.itemCount)} 个子项</span>}
            </li>
          ))}
          {assets.length > 10 && <li className="text-[10px] text-muted-foreground/60 pl-3">…还有 {assets.length - 10} 项</li>}
        </ul>
      )}
    </div>
  );
}

/** 元数据结果 — query_asset_metadata */
function MetadataResult({ data }: { data: unknown }) {
  const obj = data as Obj;
  const props =
    (obj.fields as Array<Obj>) ??
    (obj.properties as Array<Obj>) ??
    (obj.attributes as Array<Obj>);
  if (!Array.isArray(props)) return <GenericResult data={data} toolName="query_asset_metadata" />;
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">共 <span className="font-medium text-foreground">{props.length}</span> 个属性</p>
      <ul className="space-y-0.5">
        {props.slice(0, 15).map((prop, i) => (
          <li key={i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
            <span className="w-1 h-1 rounded-full bg-muted-foreground/40 shrink-0" />
            <span className="font-medium text-foreground">
              {String(prop.fieldLabel ?? prop.fieldKey ?? prop.name ?? prop.key ?? `属性${i + 1}`)}
            </span>
            {!!(prop.fieldType ?? prop.type) && (
              <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">{String(prop.fieldType ?? prop.type)}</span>
            )}
            {prop.required === true && <span className="text-[10px] text-orange-400">必填</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** 批量创建结果 — batch_create_assets */
function BatchCreateResult({ data }: { data: unknown }) {
  const obj = data as Obj;
  const created = obj.created as Array<unknown> | undefined;
  const total = (obj.total as number) ?? created?.length;
  const message = obj.message as string | undefined;
  return (
    <div className="space-y-1">
      {message && <p className="text-xs text-muted-foreground">{message}</p>}
      {total != null && (
        <p className="text-xs text-muted-foreground">
          ✅ 成功创建 <span className="font-medium text-foreground">{total}</span> 项
        </p>
      )}
      {Array.isArray(created) && created.length > 0 && (
        <ul className="space-y-0.5">
          {created.slice(0, 8).map((item, i) => {
            const it = item as Obj;
            return (
              <li key={i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
                <span className="w-1 h-1 rounded-full bg-green-400/60 shrink-0" />
                <span>{String(it.name ?? it.id ?? `#${i + 1}`)}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/** 通用结果展示 */
function GenericResult({ data, toolName }: { data: unknown; toolName: string }) {
  if (typeof data !== "object" || data === null) {
    return <p className="text-xs text-muted-foreground">{formatValue(data)}</p>;
  }
  if (Array.isArray(data)) {
    return (
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground">
          返回 <span className="font-medium text-foreground">{data.length}</span> 条记录
        </p>
        {data.slice(0, 5).map((item, i) => (
          <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground/90">
            <span className="w-1 h-1 rounded-full bg-muted-foreground/40 shrink-0" />
            <span>
              {typeof item === "object" && item !== null
                ? String((item as Obj).name ?? (item as Obj).id ?? JSON.stringify(item).slice(0, 80))
                : formatValue(item)}
            </span>
          </div>
        ))}
        {data.length > 5 && <p className="text-[10px] text-muted-foreground/60 pl-3">…还有 {data.length - 5} 条</p>}
      </div>
    );
  }

  const obj = data as Obj;
  const priorityKeys = ["status", "message", "total", "id", "name", "type", "count"];
  const entries = Object.entries(obj).sort((a, b) => {
    const ai = priorityKeys.indexOf(a[0]);
    const bi = priorityKeys.indexOf(b[0]);
    if (ai !== -1 && bi !== -1) return ai - bi;
    if (ai !== -1) return -1;
    if (bi !== -1) return 1;
    return 0;
  });

  const imgEntries = entries.filter(([, v]) => isImageUrl(v));
  const textEntries = entries.filter(([, v]) => !isImageUrl(v) && !(typeof v === "string" && v.length > 200));

  return (
    <div className="space-y-1.5">
      {imgEntries.map(([key, val]) => {
        const resolved = resolveMediaUrl(val as string);
        return resolved ? (
          <ThumbImage
            key={key}
            src={resolved}
            label={getFieldLabel(toolName, key)}
            previewClassName="h-[104px] w-[184px] sm:h-[116px] sm:w-[206px]"
          />
        ) : null;
      })}
      {textEntries.slice(0, 8).map(([key, val]) => (
        <div key={key} className="flex items-baseline gap-2 text-xs">
          <span className="text-muted-foreground/70 shrink-0">{getFieldLabel(toolName, key)}:</span>
          <span className="text-muted-foreground">
            {key === "status" ? friendlyStatus(val) : formatValue(val)}
          </span>
        </div>
      ))}
      {textEntries.length > 8 && (
        <p className="text-[10px] text-muted-foreground/60">…还有 {textEntries.length - 8} 个字段</p>
      )}
    </div>
  );
}

// ========== 入口组件（按工具名路由）==========

export function ToolResultDisplay({ toolName, result }: { toolName: string; result: string }) {
  let parsed: unknown;
  try {
    parsed = JSON.parse(result);
  } catch {
    return (
      <p className="text-xs text-muted-foreground whitespace-pre-wrap">
        {result.length > 500 ? result.slice(0, 500) + "…" : result}
      </p>
    );
  }

  // 错误统一展示
  if (typeof parsed === "object" && parsed !== null && (parsed as Obj).status === "error") {
    const msg = (parsed as Obj).message;
    return <p className="text-xs text-destructive">❌ {typeof msg === "string" ? msg : "操作失败"}</p>;
  }

  switch (toolName) {
    case "get_generation_model_capabilities":
      return <GenerationModelCapabilitiesResult data={parsed} />;
    case "generate_image":
      return <ImageGenerateResult data={parsed} />;
    case "generate_video":
      return <VideoGenerateResult data={parsed} />;
    case "list_project_assets":
      return <AssetListResult data={parsed} />;
    case "query_asset_metadata":
      return <MetadataResult data={parsed} />;
    case "batch_create_assets":
      return <BatchCreateResult data={parsed} />;
    case "update_asset_image":
    case "add_asset_item":
    case "save_script_episode":
    case "save_script_scene_items":
    case "update_script_info":
    case "save_project":
    case "delete_project":
    case "update_asset_item":
    case "delete_asset_resource":
    case "delete_script_child_resource":
    case "update_storyboard_scene":
    case "update_storyboard_item":
    case "update_storyboard_item_frame":
    case "update_storyboard_item_video":
    case "delete_storyboard_child_resource":
    case "update_script_scene":
    case "manage_script_scenes":
    case "update_script":
    case "update_asset":
      return <ToolMutationResult data={parsed} toolName={toolName} />;
    default:
      return <GenericResult data={parsed} toolName={toolName} />;
  }
}
