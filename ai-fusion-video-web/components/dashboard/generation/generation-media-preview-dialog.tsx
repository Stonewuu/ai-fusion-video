"use client";

import {
  useRef,
  useState,
  type PointerEvent,
  type SyntheticEvent,
  type WheelEvent,
} from "react";
import { Maximize2, X, ZoomIn, ZoomOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { SafeImage } from "@/components/ui/safe-image";
import { cn } from "@/lib/utils";

export interface GenerationMediaPreview {
  type: "image" | "video";
  url: string;
  title: string;
  posterUrl?: string;
}

interface GenerationMediaPreviewDialogProps {
  preview: GenerationMediaPreview | null;
  onOpenChange: (open: boolean) => void;
}

const MIN_ZOOM = 0.05;
const MAX_ZOOM = 5;
const ZOOM_STEP = 0.1;

export function GenerationMediaPreviewDialog({
  preview,
  onOpenChange,
}: GenerationMediaPreviewDialogProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const dragStateRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    scrollLeft: number;
    scrollTop: number;
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [fitZoom, setFitZoom] = useState(1);
  const [mediaSize, setMediaSize] = useState({ width: 0, height: 0 });
  const [dragging, setDragging] = useState(false);

  const changeZoom = (delta: number) => {
    setZoom((current) =>
      Math.round(
        Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current + delta)) * 100,
      ) / 100,
    );
  };

  const changeZoomByFactor = (factor: number) => {
    setZoom((current) =>
      Math.round(
        Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current * factor)) * 1000,
      ) / 1000,
    );
  };

  const initializeMediaSize = (width: number, height: number) => {
    const viewport = viewportRef.current;
    const availableWidth = Math.max((viewport?.clientWidth || width) - 32, 1);
    const availableHeight = Math.max((viewport?.clientHeight || height) - 128, 1);
    const nextFitZoom = Math.min(
      1,
      availableWidth / Math.max(width, 1),
      availableHeight / Math.max(height, 1),
    );
    setMediaSize({ width, height });
    setFitZoom(nextFitZoom);
    setZoom(nextFitZoom);
  };

  const handleImageLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    initializeMediaSize(
      event.currentTarget.naturalWidth,
      event.currentTarget.naturalHeight,
    );
  };

  const handleVideoMetadata = (event: SyntheticEvent<HTMLVideoElement>) => {
    initializeMediaSize(
      event.currentTarget.videoWidth,
      event.currentTarget.videoHeight,
    );
  };

  const handleWheel = (event: WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    changeZoomByFactor(event.deltaY < 0 ? 1.1 : 0.9);
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (preview?.type !== "image" || event.button !== 0 || zoom <= fitZoom) {
      return;
    }
    const viewport = event.currentTarget;
    event.preventDefault();
    viewport.setPointerCapture(event.pointerId);
    dragStateRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: viewport.scrollLeft,
      scrollTop: viewport.scrollTop,
    };
    setDragging(true);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    event.preventDefault();
    event.currentTarget.scrollLeft =
      dragState.scrollLeft - (event.clientX - dragState.startX);
    event.currentTarget.scrollTop =
      dragState.scrollTop - (event.clientY - dragState.startY);
  };

  const finishDragging = (event: PointerEvent<HTMLDivElement>) => {
    if (dragStateRef.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragStateRef.current = null;
    setDragging(false);
  };

  const renderedWidth = mediaSize.width > 0 ? mediaSize.width * zoom : undefined;
  const renderedHeight = mediaSize.height > 0 ? mediaSize.height * zoom : undefined;

  return (
    <Dialog
      open={Boolean(preview)}
      onOpenChange={(open) => {
        if (!open) {
          setZoom(1);
          setFitZoom(1);
          setMediaSize({ width: 0, height: 0 });
          dragStateRef.current = null;
          setDragging(false);
        }
        onOpenChange(open);
      }}
    >
      <DialogContent
        data-variant="media-lightbox"
        showCloseButton={false}
        className="media-lightbox inset-0 h-dvh w-dvw max-w-none translate-x-0 translate-y-0 overflow-hidden rounded-none border-0 p-0 sm:max-w-none"
      >
        <DialogHeader className="pointer-events-none absolute inset-x-3 top-3 z-20 flex-row items-center justify-between gap-3 sm:inset-x-4 sm:top-4">
          <div className="pointer-events-auto hidden min-w-0 max-w-[min(24rem,40vw)] rounded-2xl border border-border/40 bg-popover/82 px-4 py-3 text-popover-foreground shadow-xl backdrop-blur-xl sm:block">
            <DialogTitle className="truncate text-sm">
              {preview?.title || "媒体预览"}
            </DialogTitle>
          </div>
          <div className="pointer-events-auto ml-auto flex shrink-0 items-center gap-2 rounded-2xl border border-border/40 bg-popover/82 p-1.5 text-popover-foreground shadow-xl backdrop-blur-xl">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => changeZoom(-ZOOM_STEP)}
              disabled={zoom <= MIN_ZOOM}
              title="缩小"
              aria-label="缩小预览"
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="xs"
              onClick={() => setZoom(1)}
              title="按原始像素尺寸显示"
            >
              {Math.round(zoom * 100)}%
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setZoom(fitZoom)}
              title="适应窗口"
              aria-label="适应窗口"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => changeZoom(ZOOM_STEP)}
              disabled={zoom >= MAX_ZOOM}
              title="放大"
              aria-label="放大预览"
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
            <span className="mx-0.5 h-5 w-px bg-border/40" />
            <DialogClose
              render={
                <Button
                  variant="ghost"
                  size="icon-sm"
                  title="关闭预览"
                  aria-label="关闭预览"
                />
              }
            >
              <X className="h-4 w-4" />
            </DialogClose>
          </div>
          <DialogDescription className="sr-only">
            可在当前页面放大、缩小并查看生成媒体。
          </DialogDescription>
        </DialogHeader>

        <div
          ref={viewportRef}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishDragging}
          onPointerCancel={finishDragging}
          className={cn(
            "absolute inset-0 overflow-auto overscroll-contain",
            preview?.type === "image" && zoom > fitZoom &&
              "touch-none cursor-grab select-none",
            dragging && "cursor-grabbing",
          )}
        >
          <div
            className="grid min-h-full min-w-full place-items-center"
            style={{
              width: renderedWidth ? `${renderedWidth}px` : undefined,
              height: renderedHeight ? `${renderedHeight}px` : undefined,
            }}
          >
            {preview?.type === "video" ? (
              <video
                key={preview.url}
                src={preview.url}
                poster={preview.posterUrl}
                controls
                autoPlay
                playsInline
                onLoadedMetadata={handleVideoMetadata}
                className="max-w-none shrink-0 rounded-xl bg-black object-contain shadow-2xl ring-1 ring-border/30"
                style={{
                  width: renderedWidth ? `${renderedWidth}px` : "auto",
                  height: renderedHeight ? `${renderedHeight}px` : "auto",
                }}
              />
            ) : preview ? (
              <SafeImage
                src={preview.url}
                alt={preview.title}
                onLoad={handleImageLoad}
                draggable={false}
                className="max-w-none shrink-0 rounded-xl object-contain shadow-2xl ring-1 ring-border/30"
                style={{
                  width: renderedWidth ? `${renderedWidth}px` : "auto",
                  height: renderedHeight ? `${renderedHeight}px` : "auto",
                }}
                fallbackType="image"
              />
            ) : null}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
