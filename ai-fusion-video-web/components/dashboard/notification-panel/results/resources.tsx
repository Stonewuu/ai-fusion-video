"use client";

import { CheckCircle2, XCircle } from "lucide-react";
import { ToolMutationResult } from "@/components/dashboard/shared/tool-mutation-result";
import { assetTypeNames } from "../constants";
import { GenericResult } from "./shared";

export function AssetListResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const assets = (obj.assets as Array<Record<string, unknown>>) || [];
  const total = (obj.total as number) ?? assets.length;
  const typeStr = obj.type as string;

  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">
        共 <span className="font-medium text-foreground">{total}</span> 项
        {typeStr && typeStr !== "all" && (
          <span className="ml-1 px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 text-[10px]">
            {assetTypeNames[typeStr] || typeStr}
          </span>
        )}
      </p>
      {assets.length > 0 && (
        <ul className="space-y-1">
          {assets.slice(0, 10).map((asset, index) => (
            <li
              key={asset.id ? String(asset.id) : index}
              className="flex items-center gap-2 text-xs text-muted-foreground/90"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400/40 shrink-0" />
              <span className="font-medium text-foreground">
                {String(asset.name || "未命名")}
              </span>
              {!!asset.type && (
                <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                  {assetTypeNames[String(asset.type)] || String(asset.type)}
                </span>
              )}
              {asset.itemCount !== undefined && (
                <span className="text-[10px]">{String(asset.itemCount)} 个子项</span>
              )}
            </li>
          ))}
          {assets.length > 10 && (
            <li className="text-[10px] text-muted-foreground/60 pl-3">
              …还有 {assets.length - 10} 项
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

export function MetadataResult({ data }: { data: unknown }) {
  if (typeof data !== "object" || data === null) {
    return <GenericResult data={data} />;
  }

  const obj = data as Record<string, unknown>;
  const assetType = obj.assetType as string | undefined;
  const properties =
    (obj.fields as Array<Record<string, unknown>>) ||
    (obj.properties as Array<Record<string, unknown>>) ||
    (obj.attributes as Array<Record<string, unknown>>);

  if (!properties || !Array.isArray(properties)) {
    return <GenericResult data={data} />;
  }

  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">
        {assetType && (
          <span className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-400 text-[10px] mr-2">
            {assetTypeNames[assetType] || assetType}
          </span>
        )}
        共 <span className="font-medium text-foreground">{properties.length}</span> 个属性
      </p>
      <ul className="space-y-0.5">
        {properties.slice(0, 15).map((prop, index) => (
          <li
            key={index}
            className="flex items-center gap-2 text-xs text-muted-foreground/90"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-violet-400/40 shrink-0" />
            <span className="font-medium text-foreground">
              {String(
                prop.fieldLabel ||
                  prop.fieldKey ||
                  prop.name ||
                  prop.key ||
                  `属性${index + 1}`
              )}
            </span>
            {!!(prop.fieldType || prop.type) && (
              <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                {String(prop.fieldType || prop.type)}
              </span>
            )}
            {prop.required === true && (
              <span className="text-[10px] text-orange-400">必填</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function BatchCreateResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const created = obj.created as Array<unknown> | undefined;
  const existing = obj.existing as Array<unknown> | undefined;
  const message = obj.message as string | undefined;
  const createdCount = (obj.createdCount as number) ?? created?.length;
  const existingCount = (obj.existingCount as number) ?? existing?.length;

  return (
    <div className="space-y-2">
      {message && <p className="text-xs text-muted-foreground">{message}</p>}
      <div className="flex flex-wrap gap-1.5">
        {createdCount !== undefined && createdCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-500/10 text-green-500 text-[10px] font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
            新建 {createdCount} 项
          </span>
        )}
        {existingCount !== undefined && existingCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 text-[10px] font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            复用 {existingCount} 项
          </span>
        )}
      </div>
      {created && Array.isArray(created) && created.length > 0 && (
        <ul className="space-y-0.5">
          {created.slice(0, 8).map((item, index) => {
            const current = item as Record<string, unknown>;
            return (
              <li
                key={index}
                className="flex items-center gap-2 text-xs text-muted-foreground/90"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-green-400/60 shrink-0" />
                <span className="font-medium text-foreground">
                  {String(current.name || current.id || `#${index + 1}`)}
                </span>
                {!!current.type && (
                  <span className="px-1.5 py-0.5 rounded bg-muted/50 text-[10px]">
                    {assetTypeNames[String(current.type)] || String(current.type)}
                  </span>
                )}
              </li>
            );
          })}
          {created.length > 8 && (
            <li className="text-[10px] text-muted-foreground/60 pl-3">
              …还有 {created.length - 8} 项
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

export function MutationResult({
  data,
  toolName,
}: {
  data: unknown;
  toolName: string;
}) {
  return <ToolMutationResult data={data} toolName={toolName} />;
}

export function AssetItemsResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const assets = obj.assets as Array<Record<string, unknown>> | undefined;

  if (assets && Array.isArray(assets)) {
    const totalAssets = (obj.totalAssets as number) ?? assets.length;
    const totalItems = assets.reduce((sum, asset) => {
      const assetItems = asset.items as Array<Record<string, unknown>> | undefined;
      const assetTotalItems =
        typeof asset.totalItems === "number"
          ? asset.totalItems
          : assetItems?.length ?? 0;
      return sum + assetTotalItems;
    }, 0);

    return (
      <div className="space-y-1.5">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-medium text-foreground">子资产列表</span>
          <span className="text-[10px] text-muted-foreground/50">
            共 {totalAssets} 个资产
          </span>
          <span className="text-[10px] text-muted-foreground/50">
            共 {totalItems} 个子资产
          </span>
        </div>

        <ul className="space-y-0.5">
          {assets.slice(0, 8).map((asset, index) => {
            const status = asset.status as string | undefined;
            const message = asset.message as string | undefined;
            const assetType =
              asset.assetType == null ? undefined : String(asset.assetType);
            const assetItems = asset.items as Array<Record<string, unknown>> | undefined;
            const assetTotalItems =
              typeof asset.totalItems === "number"
                ? asset.totalItems
                : assetItems?.length ?? 0;
            const previewNames = assetItems
              ?.slice(0, 3)
              .map((item) => String(item.name || item.id || "未命名子资产"))
              .join(" / ");

            return (
              <li
                key={asset.assetId ? String(asset.assetId) : index}
                className="flex items-center gap-2 text-xs text-muted-foreground/90"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-teal-400/40 shrink-0" />
                <span className="font-medium text-foreground">
                  {String(asset.assetName || `资产${index + 1}`)}
                </span>
                {assetType ? (
                  <span className="px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 text-[10px]">
                    {assetTypeNames[assetType] || assetType}
                  </span>
                ) : null}
                {status === "error" ? (
                  <span className="text-[10px] text-destructive">
                    {message || "查询失败"}
                  </span>
                ) : (
                  <>
                    <span className="text-[10px] text-muted-foreground/50">
                      {assetTotalItems} 个子资产
                    </span>
                    {previewNames && (
                      <span className="text-[10px] text-muted-foreground/60 truncate">
                        {previewNames}
                      </span>
                    )}
                  </>
                )}
              </li>
            );
          })}
          {assets.length > 8 && (
            <li className="text-[10px] text-muted-foreground/50 pl-3">
              …还有 {assets.length - 8} 个资产
            </li>
          )}
        </ul>
      </div>
    );
  }

  const assetName = obj.assetName as string | undefined;
  const assetType = obj.assetType as string | undefined;
  const totalItems = obj.totalItems as number | undefined;
  const items = obj.items as Array<Record<string, unknown>> | undefined;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-foreground">
          {assetName || "子资产列表"}
        </span>
        {assetType && (
          <span className="px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 text-[10px]">
            {assetTypeNames[assetType] || assetType}
          </span>
        )}
        <span className="text-[10px] text-muted-foreground/50">
          共 {totalItems ?? 0} 个子资产
        </span>
      </div>
      {items && Array.isArray(items) && items.length > 0 && (
        <ul className="space-y-0.5">
          {items.slice(0, 8).map((item, index) => (
            <li
              key={item.id ? String(item.id) : index}
              className="flex items-center gap-2 text-xs text-muted-foreground/90"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-teal-400/40 shrink-0" />
              <span className="font-medium text-foreground">
                {String(item.name || `变体${index + 1}`)}
              </span>
              {item.itemType != null && (
                <span className="text-[10px] text-muted-foreground/50">
                  {String(item.itemType)}
                </span>
              )}
            </li>
          ))}
          {items.length > 8 && (
            <li className="text-[10px] text-muted-foreground/50 pl-3">
              …还有 {items.length - 8} 个
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

export function ProjectInfoResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const name = obj.name as string | undefined;
  const description = obj.description as string | undefined;

  return (
    <div className="space-y-1.5">
      <span className="text-xs font-medium text-foreground">
        {name || "未命名项目"}
      </span>
      {description && (
        <p className="text-[10px] text-muted-foreground/70 leading-relaxed">
          {description.length > 200 ? `${description.slice(0, 200)}…` : description}
        </p>
      )}
    </div>
  );
}

export function SaveEpisodeResult({ data }: { data: unknown }) {
  const obj = data as Record<string, unknown>;
  const message = obj.message as string | undefined;
  const episodeNumber = obj.episodeNumber as number | undefined;
  const title = obj.title as string | undefined;
  const episodeId = obj.scriptEpisodeId ?? obj.episodeId;
  const version = obj.episode_version;

  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
        <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
        {message || `第${episodeNumber ?? "?"}集「${title || ""}」保存成功`}
      </p>
      <div className="flex gap-3 text-[10px] text-muted-foreground/60">
        {episodeId !== undefined && <span>分集ID: {String(episodeId)}</span>}
        {version !== undefined && <span>版本: v{String(version)}</span>}
      </div>
    </div>
  );
}

export function CreateResult({
  data,
  toolName,
}: {
  data: unknown;
  toolName: string;
}) {
  const obj = data as Record<string, unknown>;
  const status = obj.status as string | undefined;
  const message = obj.message as string | undefined;
  const id = obj.id ?? obj.assetId ?? obj.itemId;
  const name = obj.name as string | undefined;

  const toolLabels: Record<string, string> = {
    create_asset: "资产",
    add_asset_item: "子资产",
    batch_create_asset_items: "批量子资产",
  };
  const label = toolLabels[toolName] || "资源";

  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground inline-flex items-center gap-1">
        {status === "error" ? (
          <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />
        ) : (
          <CheckCircle2 className="h-3.5 w-3.5 text-green-500 shrink-0" />
        )}
        {message || (status === "error" ? `${label}创建失败` : `${label}创建成功`)}
      </p>
      {name && <p className="text-[10px] text-foreground/80">名称: {name}</p>}
      {id !== undefined && (
        <p className="text-[10px] text-muted-foreground/60">
          {label}ID: {String(id)}
        </p>
      )}
    </div>
  );
}
