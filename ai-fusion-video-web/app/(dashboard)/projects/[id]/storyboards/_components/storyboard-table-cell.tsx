"use client";
import { useCallback, type ComponentProps } from "react";
import { Plus, Video, Play, ZoomIn } from "lucide-react";
import { ManagedVideo } from "@/components/media/managed-video";
import { SafeImage } from "@/components/media/managed-image";
import { Tooltip, TooltipTrigger } from "@/components/ui/tooltip";
import { FixedTooltipContent as TooltipContent } from "@/components/shared/fixed-tooltip-content";
import { resolveMediaUrl } from "@/lib/api/client";
import { cn } from "@/lib/utils";
import type { StoryboardItem, StoryboardFrameType } from "@/lib/api/storyboard";
import { EditableCell } from "./editable-cell";
import type { StoryboardTableView } from "./storyboard-table-view";
import { STORYBOARD_TABLE_COLUMNS } from "./storyboard-table-columns";
export const TABLE_TOOLTIP_CLASS =
  "flex max-w-[220px] flex-col items-start gap-0.5 text-left text-[11px]";
const TOOLTIP_CONTENT_CLASS = TABLE_TOOLTIP_CLASS;
/** 安全解析 ID 数组 */
function parseIds(raw: number[] | null | undefined): number[] {
  return raw ?? [];
}
/** 格式化资产显示名称：子变体名 (主资产名) */
function getAssetDisplayName(
  subItemName: string | null | undefined,
  parentAssetName: string,
) {
  const subName = subItemName?.trim();
  const parentName = parentAssetName.trim();
  if (
    !subName ||
    subName === "默认" ||
    subName === "默认变体" ||
    subName === "初始" ||
    subName === parentName
  ) {
    return parentName;
  }
  return `${subName} (${parentName})`;
}

type ViewProps = ComponentProps<typeof StoryboardTableView>;
export function StoryboardTableCell({
  item,
  idx,
  col,
  assetLookup,
  onSelectItem,
  onUpdateItemField,
  onOpenFrameDialog,
  onEditAssets,
  onEditingChange,
  setPreviewImageUrl,
  setPreviewImageTitle,
  setPreviewVideoUrl,
}: Pick<
  ViewProps,
  "onSelectItem" | "onUpdateItemField" | "onOpenFrameDialog" | "onEditAssets"
> & {
  item: StoryboardItem;
  idx: number;
  col: (typeof STORYBOARD_TABLE_COLUMNS)[number];
  assetLookup: NonNullable<ViewProps["assetLookup"]>;
  setPreviewImageUrl: (url: string) => void;
  setPreviewImageTitle: (title: string) => void;
  setPreviewVideoUrl: (url: string) => void;
  onEditingChange: (itemId: number, field: string, editing: boolean) => void;
}) {
  const handleEditingChange = useCallback(
    (editing: boolean) => onEditingChange(item.id, col.field, editing),
    [item.id, col.field, onEditingChange],
  );
  const framePreviewUrl =
    item.firstFrameImageUrl ||
    item.generatedImageUrl ||
    item.imageUrl ||
    item.referenceImageUrl;
  return (
    <div
      key={col.field}
      data-storyboard-field={col.field}
      className="px-2 py-2 flex items-center justify-center min-w-0"
    >
      {col.isImage ? (
        <div
          onClick={(e) => {
            if (framePreviewUrl) {
              e.stopPropagation();
              setPreviewImageUrl(framePreviewUrl);
              setPreviewImageTitle(
                `镜头 #${item.shotNumber || item.autoShotNumber || ""} 画面`,
              );
            }
          }}
          className={cn(
            "flex items-center justify-center h-11 w-16 rounded-md bg-muted/20 border border-border/10 overflow-hidden shrink-0 relative group/img",
            framePreviewUrl &&
              "cursor-zoom-in hover:border-primary/40 transition-colors",
          )}
        >
          <SafeImage
            src={resolveMediaUrl(framePreviewUrl)}
            alt="画面"
            fallbackType="image"
            className="w-full h-full object-cover transition-transform group-hover/img:scale-105"
          />
          {framePreviewUrl && (
            <div className="absolute inset-0 bg-black/0 group-hover/img:bg-black/25 flex items-center justify-center opacity-0 group-hover/img:opacity-100 transition-all pointer-events-none">
              <ZoomIn className="h-3.5 w-3.5 text-white/90" />
            </div>
          )}
        </div>
      ) : col.field === "frameReferences" ? (
        <div className="flex items-center justify-center gap-1">
          {(["first", "last"] as StoryboardFrameType[]).map((frameType) => {
            const label = frameType === "first" ? "首" : "尾";
            const title = frameType === "first" ? "首帧参考图" : "尾帧参考图";
            const hasFrame =
              frameType === "first"
                ? !!item.firstFrameImageUrl
                : !!item.lastFrameImageUrl;
            return (
              <Tooltip key={frameType}>
                <TooltipTrigger
                  render={
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectItem(item.id);
                        onOpenFrameDialog?.(item, frameType);
                      }}
                      disabled={!onOpenFrameDialog}
                      className={cn(
                        "h-7 w-8 rounded-md border text-[10px] font-semibold transition-all",
                        "flex items-center justify-center disabled:opacity-45 disabled:cursor-not-allowed",
                        hasFrame
                          ? "border-emerald-500/35 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                          : "border-border/30 bg-muted/20 text-muted-foreground hover:border-primary/40 hover:text-primary hover:bg-primary/5",
                      )}
                    >
                      {label}
                    </button>
                  }
                />
                <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                  {title}：{hasFrame ? "已设置" : "未设置"}
                </TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      ) : col.isVideo ? (
        <div className="flex items-center justify-center h-11 w-16 rounded-md bg-muted/20 border border-border/10 overflow-hidden shrink-0 relative group/video">
          {item.generatedVideoUrl || item.videoUrl ? (
            <>
              <ManagedVideo
                src={
                  resolveMediaUrl(item.generatedVideoUrl || item.videoUrl) || ""
                }
                className="w-full h-full object-cover"
                muted
                preload="metadata"
                playsInline
              />
              <Tooltip>
                <TooltipTrigger
                  render={
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectItem(item.id);
                        const rawVideoUrl =
                          item.generatedVideoUrl || item.videoUrl;
                        if (rawVideoUrl) {
                          setPreviewVideoUrl(rawVideoUrl);
                        }
                      }}
                      className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover/video:bg-black/40 transition-colors"
                    >
                      <Play className="h-3.5 w-3.5 text-white/90 fill-white/90" />
                    </button>
                  }
                />
                <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                  预览视频
                </TooltipContent>
              </Tooltip>
            </>
          ) : (
            <Video className="h-3.5 w-3.5 text-muted-foreground/30" />
          )}
        </div>
      ) : col.field === "assets" ? (
        <div
          onClick={(e) => {
            e.stopPropagation();
            onSelectItem(item.id);
            onEditAssets?.(item);
          }}
          className="group/cell w-full h-full min-h-11 flex items-center justify-center px-1.5 py-1.5 rounded-lg border border-transparent hover:border-primary/20 hover:bg-primary/5 transition-all duration-200 cursor-pointer"
        >
          {(() => {
            const charIds = parseIds(item.characterIds);
            const sceneId = item.sceneAssetItemId;
            const propIds = parseIds(item.propIds);
            const charItems = charIds
              .map((id) => assetLookup[id])
              .filter(Boolean);
            const sceneItem = sceneId ? assetLookup[sceneId] : null;
            const propItems = propIds
              .map((id) => assetLookup[id])
              .filter(Boolean);
            const hasAssets =
              charItems.length > 0 || !!sceneItem || propItems.length > 0;
            if (!hasAssets) {
              return (
                <span className="text-[10px] text-muted-foreground/35 group-hover/cell:text-primary transition-colors flex items-center gap-1 font-medium select-none">
                  + 关联资产
                </span>
              );
            }
            return (
              <div className="flex items-center justify-center w-full py-1">
                <div className="flex items-center justify-start gap-2.5 max-w-full overflow-x-auto flex-nowrap [&::-webkit-scrollbar]:h-1 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-muted-foreground/20 [&::-webkit-scrollbar-thumb]:rounded-full hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/35 pb-1.5 -mb-0.5 px-1.5">
                  {/* 角色 */}
                  {charItems.map((ci, idx) => (
                    <Tooltip key={`char-${idx}`}>
                      <TooltipTrigger
                        render={
                          <div className="flex flex-col items-center gap-1 group/asset cursor-pointer select-none shrink-0">
                            <div className="relative">
                              <SafeImage
                                src={resolveMediaUrl(ci.item.imageUrl)}
                                alt="avatar"
                                fallbackType="avatar"
                                onClick={(e) => {
                                  if (ci.item.imageUrl) {
                                    e.stopPropagation();
                                    setPreviewImageUrl(ci.item.imageUrl);
                                    setPreviewImageTitle(
                                      `角色: ${getAssetDisplayName(ci.item.name, ci.asset.name)}`,
                                    );
                                  }
                                }}
                                className="h-8 w-8 rounded-full object-cover cursor-zoom-in hover:scale-110 active:scale-95 hover:shadow-md transition-all duration-200 border border-border/40 shrink-0"
                              />
                            </div>
                            <span className="text-[10px] font-semibold text-blue-500 dark:text-blue-400 truncate max-w-[56px] leading-tight text-center mt-0.5">
                              {getAssetDisplayName(ci.item.name, ci.asset.name)}
                            </span>
                          </div>
                        }
                      />
                      <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                        <span className="font-semibold">
                          角色:{" "}
                          {getAssetDisplayName(ci.item.name, ci.asset.name)}
                        </span>
                        {ci.asset.description && (
                          <span className="text-[10px] opacity-80 leading-normal break-words mt-0.5">
                            {ci.asset.description}
                          </span>
                        )}
                      </TooltipContent>
                    </Tooltip>
                  ))}
                  {/* 场景 */}
                  {sceneItem && (
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <div className="flex flex-col items-center gap-1 group/asset cursor-pointer select-none shrink-0">
                            <div className="relative">
                              <SafeImage
                                src={resolveMediaUrl(sceneItem.item.imageUrl)}
                                alt="scene"
                                fallbackType="scene"
                                onClick={(e) => {
                                  if (sceneItem.item.imageUrl) {
                                    e.stopPropagation();
                                    setPreviewImageUrl(sceneItem.item.imageUrl);
                                    setPreviewImageTitle(
                                      `场景: ${getAssetDisplayName(sceneItem.item.name, sceneItem.asset.name)}`,
                                    );
                                  }
                                }}
                                className="h-8 w-8 rounded-lg object-cover cursor-zoom-in hover:scale-110 active:scale-95 hover:shadow-md transition-all duration-200 border border-border/40 shrink-0"
                              />
                            </div>
                            <span className="text-[10px] font-semibold text-green-500 dark:text-green-400 truncate max-w-[56px] leading-tight text-center mt-0.5">
                              {getAssetDisplayName(
                                sceneItem.item.name,
                                sceneItem.asset.name,
                              )}
                            </span>
                          </div>
                        }
                      />
                      <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                        <span className="font-semibold">
                          场景:{" "}
                          {getAssetDisplayName(
                            sceneItem.item.name,
                            sceneItem.asset.name,
                          )}
                        </span>
                        {sceneItem.asset.description && (
                          <span className="text-[10px] opacity-80 leading-normal break-words mt-0.5">
                            {sceneItem.asset.description}
                          </span>
                        )}
                      </TooltipContent>
                    </Tooltip>
                  )}
                  {/* 道具 */}
                  {propItems.map((pi, idx) => (
                    <Tooltip key={`prop-${idx}`}>
                      <TooltipTrigger
                        render={
                          <div className="flex flex-col items-center gap-1 group/asset cursor-pointer select-none shrink-0">
                            <div className="relative">
                              <SafeImage
                                src={resolveMediaUrl(pi.item.imageUrl)}
                                alt="prop"
                                fallbackType="prop"
                                onClick={(e) => {
                                  if (pi.item.imageUrl) {
                                    e.stopPropagation();
                                    setPreviewImageUrl(pi.item.imageUrl);
                                    setPreviewImageTitle(
                                      `道具: ${getAssetDisplayName(pi.item.name, pi.asset.name)}`,
                                    );
                                  }
                                }}
                                className="h-8 w-8 rounded-lg object-cover cursor-zoom-in hover:scale-110 active:scale-95 hover:shadow-md transition-all duration-200 border border-border/40 shrink-0"
                              />
                            </div>
                            <span className="text-[10px] font-semibold text-amber-500 dark:text-amber-400 truncate max-w-[56px] leading-tight text-center mt-0.5">
                              {getAssetDisplayName(pi.item.name, pi.asset.name)}
                            </span>
                          </div>
                        }
                      />
                      <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                        <span className="font-semibold">
                          道具:{" "}
                          {getAssetDisplayName(pi.item.name, pi.asset.name)}
                        </span>
                        {pi.asset.description && (
                          <span className="text-[10px] opacity-80 leading-normal break-words mt-0.5">
                            {pi.asset.description}
                          </span>
                        )}
                      </TooltipContent>
                    </Tooltip>
                  ))}

                  {/* 右侧内联加号 */}
                  <Tooltip>
                    <TooltipTrigger
                      render={
                        <div className="flex flex-col items-center gap-1 group/add cursor-pointer select-none opacity-40 group-hover/cell:opacity-100 transition-opacity duration-200 ml-1 shrink-0">
                          <div className="h-8 w-8 rounded-full border border-dashed border-muted-foreground/35 bg-muted/5 group-hover/add:bg-primary/5 group-hover/add:border-primary/50 group-hover/add:text-primary flex items-center justify-center text-muted-foreground/45 transition-all duration-200 shrink-0">
                            <Plus className="h-4 w-4" />
                          </div>
                          <span className="text-[10px] font-semibold text-muted-foreground/60 group-hover/add:text-primary transition-colors leading-tight text-center mt-0.5">
                            添加
                          </span>
                        </div>
                      }
                    />
                    <TooltipContent className={TOOLTIP_CONTENT_CLASS}>
                      添加或编辑关联资产
                    </TooltipContent>
                  </Tooltip>
                </div>
              </div>
            );
          })()}
        </div>
      ) : (
        <EditableCell
          value={
            col.field === "shotNumber"
              ? item.shotNumber || String(idx + 1)
              : col.field === "duration"
                ? item.duration
                  ? String(item.duration)
                  : ""
                : (item[col.field as keyof StoryboardItem] as string) || ""
          }
          placeholder={col.field === "shotNumber" ? String(idx + 1) : ""}
          onSave={(val) => onUpdateItemField(item, col.field, val)}
          onCellClick={() => onSelectItem(item.id)}
          onEditingChange={handleEditingChange}
          className={cn(
            "text-[11px] w-full",
            col.field === "shotNumber"
              ? "font-mono text-muted-foreground text-center"
              : col.multiline
                ? "text-left"
                : "text-muted-foreground text-center",
          )}
          multiline={col.multiline}
          singleLine={col.singleLine}
        />
      )}
    </div>
  );
}
