"use client";

import { useEffect, useRef, useState } from "react";
import { Clapperboard, Film } from "lucide-react";
import { SafeImage } from "@/components/ui/safe-image";
import { resolveMediaUrl } from "@/lib/api/client";
import { storyboardShotTypeNames } from "../constants";

type StoryboardShot = Record<string, unknown>;

const COLLAPSED_SHOT_COUNT = 3;
const EXPANDED_SHOT_BATCH = 24;

function StoryboardShotCard({
  item,
  index,
}: {
  item: StoryboardShot;
  index: number;
}) {
  const shotNumber = item.shotNumber ?? item.autoShotNumber ?? index + 1;
  const sceneNumber = typeof item.sceneNumber === "string"
    ? item.sceneNumber.trim()
    : "";
  // Historical tool results do not contain sceneNumber, so retain their
  // original shot label while new results use the unambiguous scene-shot form.
  const displayNumber = sceneNumber
    ? `${sceneNumber}-${String(shotNumber)}`
    : String(shotNumber);
  const shotType = item.shotType as string | undefined;
  const cameraMovement = item.cameraMovement as string | undefined;
  const content =
    (item.content as string | undefined)
    || (item.sceneExpectation as string | undefined)
    || "（无画面描述）";
  const duration = typeof item.duration === "number" ? item.duration : undefined;
  const imageUrl = resolveMediaUrl(
    (item.generatedImageUrl as string | null | undefined)
      || (item.imageUrl as string | null | undefined),
  );
  const videoUrl = resolveMediaUrl(
    (item.generatedVideoUrl as string | null | undefined)
      || (item.videoUrl as string | null | undefined),
  );
  const hasImage = !!imageUrl;
  const hasVideo = !!videoUrl;

  return (
    <div
      className="rounded-lg border border-border/25 bg-background/50 px-2 py-1.5"
      style={{ contentVisibility: "auto", containIntrinsicSize: "88px" }}
    >
      <div className="flex gap-2">
        <div className="relative min-h-10 w-16 shrink-0 self-stretch overflow-hidden rounded-md border border-border/20 bg-muted/20">
          {hasVideo ? (
            <>
              <video
                src={videoUrl || ""}
                className="absolute inset-0 h-full w-full object-contain"
                muted
                playsInline
                preload="none"
              />
              <div className="absolute inset-0 flex items-center justify-center bg-black/15">
                <Clapperboard className="h-3.5 w-3.5 text-white/90" />
              </div>
            </>
          ) : hasImage ? (
            <SafeImage
              src={imageUrl || ""}
              alt={`镜头 ${displayNumber}`}
              className="absolute inset-0 h-full w-full object-contain"
              loading="lazy"
              decoding="async"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <Film className="h-4 w-4 text-muted-foreground/35" />
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1 space-y-0.5">
          <div className="flex flex-wrap items-center gap-1">
            <span className="text-xs font-semibold tabular-nums text-foreground">
              {displayNumber}
            </span>
            {shotType ? (
              <span className="rounded bg-muted/50 px-1.5 py-0.5 text-[10px] text-muted-foreground">
                {storyboardShotTypeNames[shotType] || shotType}
              </span>
            ) : null}
            {cameraMovement ? (
              <span className="rounded bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-400">
                {cameraMovement}
              </span>
            ) : null}
            {duration !== undefined ? (
              <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400">
                {duration}s
              </span>
            ) : null}
          </div>

          <p className="line-clamp-2 text-[10px] leading-4 text-muted-foreground/85">
            {content}
          </p>

          <div className="flex flex-wrap items-center gap-1">
            {hasImage ? (
              <span className="text-[10px] text-cyan-400/80">有画面</span>
            ) : null}
            {hasVideo ? (
              <span className="text-[10px] text-emerald-400/80">已有视频</span>
            ) : null}
            {!hasImage && !hasVideo ? (
              <span className="text-[10px] text-muted-foreground/60">暂无媒体</span>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

export function StoryboardShotList({
  items,
  expanded,
}: {
  items: StoryboardShot[];
  expanded: boolean;
}) {
  const [visibleCount, setVisibleCount] = useState(() => Math.min(
    items.length,
    expanded ? EXPANDED_SHOT_BATCH : COLLAPSED_SHOT_COUNT,
  ));
  const sentinelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!expanded || visibleCount >= items.length) return;
    const sentinel = sentinelRef.current;
    if (!sentinel) throw new Error("Expanded storyboard result has no load sentinel");

    const observer = new IntersectionObserver((entries) => {
      if (!entries[0]?.isIntersecting) return;
      setVisibleCount((current) => Math.min(
        current + EXPANDED_SHOT_BATCH,
        items.length,
      ));
    }, { rootMargin: "600px 0px" });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [expanded, items.length, visibleCount]);

  const remaining = items.length - visibleCount;

  return (
    <div className="space-y-1.5">
      {items.slice(0, visibleCount).map((item, index) => (
        <StoryboardShotCard
          key={String(item.id ?? index)}
          item={item}
          index={index}
        />
      ))}

      {remaining > 0 ? (
        <div ref={sentinelRef} className="px-1 py-1 text-[10px] text-muted-foreground/60">
          {expanded
            ? `继续滚动加载剩余 ${remaining} 个镜头`
            : `…还有 ${remaining} 个镜头`}
        </div>
      ) : expanded ? (
        <p className="px-1 py-1 text-[10px] text-muted-foreground/60">
          已显示全部 {items.length} 个镜头
        </p>
      ) : null}
    </div>
  );
}
