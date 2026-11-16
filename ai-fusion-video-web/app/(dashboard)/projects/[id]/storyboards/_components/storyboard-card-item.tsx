"use client";
import { ManagedVideo } from "@/components/media/managed-video";
import React, { memo, useState } from "react";
import {
  Clock,
  Camera,
  Image as ImageIcon,
  GripHorizontal,
  Video,
  Play,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { resolveMediaUrl } from "@/lib/api/client";
import type { StoryboardFrameType, StoryboardItem } from "@/lib/api/storyboard";
import { SafeImage } from "@/components/media/managed-image";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
type SortableBindings = {
  attributes?: ReturnType<typeof useSortable>["attributes"];
  listeners?: ReturnType<typeof useSortable>["listeners"];
};
// 使用 React.memo 避免拖拽时其他不需要参与重排的卡片重新渲染，造成卡顿
export const CardItemUI = memo(
  React.forwardRef<
    HTMLDivElement,
    {
      item: StoryboardItem;
      idx: number;
      isSelected: boolean;
      onSelect?: () => void;
      onDelete?: (itemId: number) => void;
      onVideoGen?: (itemId: number) => void;
      onOpenFrameDialog?: (
        item: StoryboardItem,
        frameType: StoryboardFrameType,
      ) => void;
      onPreviewVideo?: (videoUrl: string) => void;
      attributes?: SortableBindings["attributes"];
      listeners?: SortableBindings["listeners"];
      style?: React.CSSProperties;
      isDragging?: boolean;
      isOverlay?: boolean;
    }
  >(
    (
      {
        item,
        idx,
        isSelected,
        onSelect,
        onDelete,
        onVideoGen,
        onOpenFrameDialog,
        onPreviewVideo,
        attributes,
        listeners,
        style,
        isDragging,
        isOverlay,
      },
      ref,
    ) => {
      const hasImage =
        item.firstFrameImageUrl ||
        item.imageUrl ||
        item.referenceImageUrl ||
        item.generatedImageUrl;
      const hasVideo = item.generatedVideoUrl || item.videoUrl;
      const hasBoth = hasImage && hasVideo;
      // 有视频时默认展示视频，否则展示图片
      const [mediaMode, setMediaMode] = useState<"image" | "video">(
        hasVideo ? "video" : "image",
      );
      const rawVideoUrl = (item.generatedVideoUrl ||
        item.videoUrl ||
        "") as string;
      const videoSrc =
        resolveMediaUrl(item.generatedVideoUrl || item.videoUrl) || "";
      const imageSrc = (item.firstFrameImageUrl ||
        item.generatedImageUrl ||
        item.imageUrl ||
        item.referenceImageUrl) as string;
      return (
        <div
          ref={ref}
          style={style}
          onClick={onSelect}
          className={cn(
            "rounded-xl overflow-hidden cursor-pointer relative bg-card",
            "transition-all duration-200 group flex flex-col",
            isOverlay
              ? "ring-2 ring-primary shadow-2xl scale-[1.02] cursor-grabbing z-50 border border-primary/20 bg-background"
              : isDragging
                ? "opacity-30 z-0 bg-muted/20 grayscale pointer-events-none"
                : "hover:shadow-lg hover:-translate-y-1",
            !isOverlay && !isDragging && isSelected
              ? "border-2 border-primary ring-4 ring-primary/10 shadow-md shadow-primary/10"
              : !isOverlay && !isDragging
                ? "border border-border/50 hover:border-primary/40 shadow-xs"
                : "",
          )}
        >
          {/* 画面/视频区域 16:9 */}
          <div className="aspect-video relative bg-muted/40 overflow-hidden shrink-0 border-b border-border/50">
            {/* 视频模式 */}
            {mediaMode === "video" && hasVideo ? (
              <>
                <ManagedVideo
                  src={videoSrc}
                  className="w-full h-full object-cover"
                  muted
                  preload="metadata"
                  playsInline
                />
                {onPreviewVideo && rawVideoUrl ? (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelect?.();
                      onPreviewVideo(rawVideoUrl);
                    }}
                    className="absolute inset-0 flex items-center justify-center bg-black/20 transition-colors hover:bg-black/40"
                    title="预览视频"
                  >
                    <Play className="h-8 w-8 text-white/90 fill-white/90" />
                  </button>
                ) : null}
              </>
            ) : hasImage ? (
              /* 图片模式 */
              <SafeImage
                src={resolveMediaUrl(imageSrc) || undefined}
                alt={item.content || `镜头 ${item.shotNumber || idx + 1}`}
                fallbackType="image"
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
              />
            ) : (
              /* 无内容占位 */
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <ImageIcon className="h-8 w-8 text-muted-foreground/20 mb-2" />
                <span className="text-[11px] font-medium text-muted-foreground/40">
                  暂无画面
                </span>
              </div>
            )}

            {/* 拖拽手柄 - 顶部居中悬浮 */}
            <div
              {...attributes}
              {...listeners}
              className={cn(
                "absolute top-2 left-1/2 -translate-x-1/2 p-1.5 rounded-md bg-black/40 backdrop-blur-sm opacity-0 group-hover:opacity-100 cursor-grab hover:bg-black/60 transition-all z-20",
                isOverlay && "cursor-grabbing opacity-100 bg-black/60",
                isDragging && "opacity-0",
              )}
              onClick={(e) => e.stopPropagation()}
            >
              <GripHorizontal className="h-4 w-4 text-white/90" />
            </div>

            {/* 镜号标签 -> 左上角 */}
            <div className="absolute top-2 left-2 px-1.5 py-0.5 rounded bg-black/50 backdrop-blur-sm text-white/95 text-[10px] font-bold z-10">
              #{item.shotNumber || idx + 1}
            </div>

            {/* 时长与删除操作 -> 右上角 */}
            <div className="absolute top-2 right-2 z-20 flex items-center gap-2">
              {item.duration && (
                <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/50 backdrop-blur-sm text-white/95 text-[10px]">
                  <Clock className="h-2.5 w-2.5 opacity-80" />
                  <span className="font-medium">{item.duration}s</span>
                </div>
              )}
              {onDelete && (
                <Button
                  type="button"
                  variant="destructive"
                  size="icon-sm"
                  onClick={(event) => {
                    event.stopPropagation();
                    onDelete(item.id);
                  }}
                  aria-label={`删除镜头 ${item.shotNumber || idx + 1}`}
                  title="删除镜头"
                >
                  <Trash2 />
                </Button>
              )}
            </div>

            {/* 图片/视频切换 tab - 底部居中 */}
            {hasBoth && (
              <div
                className="absolute bottom-2 left-1/2 -translate-x-1/2 flex items-center gap-0.5 p-0.5 rounded-md bg-black/50 backdrop-blur-sm z-20"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => setMediaMode("image")}
                  className={cn(
                    "flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors",
                    mediaMode === "image"
                      ? "bg-white/20 text-white"
                      : "text-white/60 hover:text-white/90",
                  )}
                >
                  <ImageIcon className="h-3 w-3" />
                  图片
                </button>
                <button
                  onClick={() => setMediaMode("video")}
                  className={cn(
                    "flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors",
                    mediaMode === "video"
                      ? "bg-white/20 text-white"
                      : "text-white/60 hover:text-white/90",
                  )}
                >
                  <Play className="h-3 w-3" />
                  视频
                </button>
              </div>
            )}

            {/* 首尾帧入口 - 每个镜头独立设置 */}
            {onOpenFrameDialog && (
              <div
                className="absolute bottom-2 left-2 flex items-center gap-1 z-20"
                onClick={(e) => e.stopPropagation()}
              >
                {(["first", "last"] as StoryboardFrameType[]).map(
                  (frameType) => {
                    const label = frameType === "first" ? "首" : "尾";
                    const title =
                      frameType === "first" ? "首帧参考图" : "尾帧参考图";
                    const hasFrame =
                      frameType === "first"
                        ? !!item.firstFrameImageUrl
                        : !!item.lastFrameImageUrl;
                    return (
                      <button
                        key={frameType}
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelect?.();
                          onOpenFrameDialog(item, frameType);
                        }}
                        className={cn(
                          "h-7 w-7 rounded-md border text-[10px] font-semibold backdrop-blur-sm transition-all",
                          "flex items-center justify-center shadow-sm",
                          hasFrame
                            ? "border-emerald-400/50 bg-emerald-500/80 text-white"
                            : "border-white/20 bg-black/45 text-white/80 hover:bg-primary/70 hover:text-white",
                        )}
                        title={`${title}：${hasFrame ? "已设置" : "未设置"}`}
                      >
                        {label}
                      </button>
                    );
                  },
                )}
              </div>
            )}

            {/* 生成视频按钮 - 右下角悬浮（只在图片模式或没有切换 tab 时显示） */}
            {onVideoGen && !hasBoth && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onVideoGen(item.id);
                }}
                className={cn(
                  "absolute bottom-2 right-2 p-1.5 rounded-md bg-black/40 backdrop-blur-sm",
                  "opacity-0 group-hover:opacity-100 transition-all z-20",
                  "hover:bg-purple-500/60 text-white/90",
                )}
                title="生成视频"
              >
                <Video className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          {/* 信息区域 (固定高度: 确保所有卡片对齐并避免网格跳动) */}
          <div className="h-[140px] p-3 flex flex-col">
            {/* 第一行：景别 & 镜头运动 */}
            <div className="flex items-center gap-1.5 flex-wrap shrink-0 mb-2">
              {item.shotType && (
                <span className="text-[10px] px-1.5 py-0.5 text-cyan-600 bg-cyan-50 dark:bg-cyan-500/10 dark:text-cyan-400 font-medium rounded-sm">
                  {item.shotType}
                </span>
              )}
              {item.cameraMovement && (
                <span className="flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 text-violet-600 bg-violet-50 dark:bg-violet-500/10 dark:text-violet-400 font-medium rounded-sm">
                  <Camera className="h-2.5 w-2.5" />
                  {item.cameraMovement}
                </span>
              )}
            </div>

            {/* 画面内容 */}
            <div className="flex-1 min-h-0">
              {item.content ? (
                <p className="text-[13px] text-foreground/90 leading-snug line-clamp-2">
                  {item.content}
                </p>
              ) : (
                <p className="text-xs text-muted-foreground/30 italic">
                  暂无画面描述
                </p>
              )}
            </div>

            {/* 对白 */}
            <div className="shrink-0 mt-2 border-t border-border/30 pt-2 h-10">
              {item.dialogue ? (
                <p className="text-xs text-muted-foreground leading-tight line-clamp-2">
                  <span className="text-primary/70 font-semibold mr-1">对</span>
                  {item.dialogue}
                </p>
              ) : (
                <div className="w-full h-full flex items-center">
                  <span className="text-[10px] text-muted-foreground/30">
                    暂无对白
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    },
  ),
);
CardItemUI.displayName = "CardItemUI";
export function SortableCardItem({
  item,
  idx,
  isSelected,
  onSelect,
  onDelete,
  onVideoGen,
  onOpenFrameDialog,
  onPreviewVideo,
}: {
  item: StoryboardItem;
  idx: number;
  isSelected: boolean;
  onSelect: () => void;
  onDelete?: (itemId: number) => void;
  onVideoGen?: (itemId: number) => void;
  onOpenFrameDialog?: (
    item: StoryboardItem,
    frameType: StoryboardFrameType,
  ) => void;
  onPreviewVideo?: (videoUrl: string) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: item.id });
  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };
  return (
    <CardItemUI
      ref={setNodeRef}
      style={style}
      item={item}
      idx={idx}
      isSelected={isSelected}
      onSelect={onSelect}
      onDelete={onDelete}
      onVideoGen={onVideoGen}
      onOpenFrameDialog={onOpenFrameDialog}
      onPreviewVideo={onPreviewVideo}
      attributes={attributes}
      listeners={listeners}
      isDragging={isDragging}
    />
  );
}
