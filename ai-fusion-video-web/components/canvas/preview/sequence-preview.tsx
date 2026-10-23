"use client";
import { useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { ManagedVideo } from "@/components/media/managed-video";
import { SafeImage } from "@/components/media/managed-image";
import { shotVideo } from "@/lib/canvas/resource-media";
import { resourceTitle, resourceKey } from "@/lib/canvas/resource-presentation";
import type { CanvasResource } from "@/lib/canvas/types";

export function CanvasSequencePreview({
  shots,
  onClose,
}: {
  shots: CanvasResource[];
  onClose: () => void;
}) {
  const [index, setIndex] = useState(0);
  const [continuous, setContinuous] = useState(false);
  const [playError, setPlayError] = useState<string | null>(null);
  const player = useRef<HTMLVideoElement>(null);
  const shot = shots[index];
  if (!shot) return null;
  const video = shotVideo(shot.data);
  const frame = shot.data.firstFrameImageUrl;
  const next = () => {
    setPlayError(null);
    setIndex((value) => Math.min(shots.length - 1, value + 1));
  };
  const play = () => {
    setPlayError(null);
    void player.current?.play().catch(() =>
      setPlayError("浏览器暂停了自动播放，请点击播放继续。"),
    );
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] sm:max-w-5xl flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {index + 1} / {shots.length} · {resourceTitle(shot)}
          </DialogTitle>
          <DialogDescription>
            按分镜中的镜头顺序预览。缺少视频时暂停，补齐素材后可继续。
          </DialogDescription>
        </DialogHeader>
        <div className="min-h-0 space-y-3 overflow-y-auto">
          {typeof video === "string" && video ? (
            <ManagedVideo
              key={`${resourceKey(shot)}:${video}`}
              ref={player}
              src={video}
              controls
              playsInline
              autoPlay={continuous}
              className="max-h-[60vh] w-full rounded-lg"
              onLoadedData={() => {
                if (continuous) play();
              }}
              onEnded={() => {
                if (continuous && index + 1 < shots.length) next();
              }}
              onError={() =>
                setPlayError("当前镜头视频无法播放，请检查素材或切换下一镜头。")
              }
            />
          ) : (
            <div className="space-y-3 rounded-lg border border-border/20 bg-background/70 p-4">
              {typeof frame === "string" && frame && (
                <SafeImage
                  src={frame}
                  alt={resourceTitle(shot)}
                  className="max-h-[50vh] w-full object-contain"
                />
              )}
              <p className="text-sm text-muted-foreground">
                当前镜头缺少视频。
              </p>
            </div>
          )}
          {playError && (
            <p role="alert" className="text-sm text-destructive">
              {playError}
            </p>
          )}
          <p className="whitespace-pre-wrap text-sm">
            {String(shot.data.content ?? "")}
          </p>
          <p className="whitespace-pre-wrap text-sm text-muted-foreground">
            {String(shot.data.dialogue ?? "")}
          </p>
        </div>
        <div
          className="flex shrink-0 flex-wrap items-center gap-2"
          role="group"
          aria-label="镜头播放控制"
        >
          <CanvasSegmentedControl
            label="播放方式"
            value={continuous ? "continuous" : "single"}
            options={[
              { value: "single", label: "单段播放" },
              { value: "continuous", label: "连续播放" },
            ]}
            className="mr-auto"
            onChange={(value) => {
              setContinuous(value === "continuous");
              if (value === "continuous") play();
            }}
          />
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              disabled={index === 0}
              onClick={() => {
                setPlayError(null);
                setIndex(index - 1);
              }}
            >
              <ChevronLeft />
              上一镜头
            </Button>
            <Button
              variant="outline"
              disabled={index + 1 === shots.length}
              onClick={next}
            >
              下一镜头
              <ChevronRight />
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
