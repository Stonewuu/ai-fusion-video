"use client";

import { useRef, useState } from "react";
import { AudioLines, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ManagedVideo } from "@/components/media/managed-video";
import { ManagedAudio } from "@/components/media/managed-audio";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";

const timeLabel = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

/** 仅底部控制区停止拖拽事件，播放中的画面仍是节点拖拽区域。 */
export function CanvasNodeInlinePlayer({ media }: { media: CanvasMediaPreview }) {
  const video = useRef<HTMLVideoElement>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [duration, setDuration] = useState(0);
  const [time, setTime] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const element = () => media.kind === "video" ? video.current : audio.current;
  const toggle = async () => {
    const player = element();
    if (!player) return;
    if (!player.paused) player.pause();
    else {
      try { await player.play(); setError(null); }
      catch { setError("播放失败，请重试或打开预览。"); }
    }
  };
  const events = {
    onPlay: () => setPlaying(true),
    onPause: () => setPlaying(false),
    onEnded: () => setPlaying(false),
    onLoadedMetadata: (event: React.SyntheticEvent<HTMLMediaElement>) => {
      const next = event.currentTarget.duration;
      setDuration(Number.isFinite(next) ? next : 0);
    },
    onTimeUpdate: (event: React.SyntheticEvent<HTMLMediaElement>) => setTime(event.currentTarget.currentTime),
    onError: () => setError("媒体暂时无法播放，请检查文件或打开预览。"),
  };
  return <div className="absolute inset-0 flex flex-col">
    <div className="flex min-h-0 flex-1 items-center justify-center">
      {media.kind === "video" ? <ManagedVideo ref={video} src={media.source} aria-label={media.title}
        playsInline preload="metadata" muted={muted} className="pointer-events-none h-full w-full object-contain" {...events} />
        : <><AudioLines className="pointer-events-none size-8 text-primary" /><ManagedAudio ref={audio} src={media.source} muted={muted} preload="metadata" {...events} /></>}
    </div>
    <div className="nodrag nopan nowheel flex shrink-0 flex-col gap-1 border-t border-border/20 bg-card p-1"
      onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
      <div className="flex min-w-0 items-center gap-2">
        <Button variant="ghost" size="icon-sm" aria-label={`${playing ? "暂停" : "播放"}${media.title}`} onClick={() => void toggle()}>{playing ? <Pause /> : <Play />}</Button>
        <input type="range" min={0} max={duration || 1} step={0.1} value={Math.min(time, duration || 0)} disabled={!duration}
          aria-label={`${media.title}播放进度`} aria-valuetext={`${timeLabel(time)} / ${timeLabel(duration)}`}
          className="h-8 min-w-0 flex-1 rounded-lg accent-primary outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
          onChange={(event) => { const player = element(); if (player) { player.currentTime = Number(event.target.value); setTime(player.currentTime); } }} />
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{timeLabel(time)} / {timeLabel(duration)}</span>
        <Button variant="ghost" size="icon-sm" aria-label={muted ? "开启声音" : "静音"} onClick={() => setMuted(!muted)}>{muted ? <VolumeX /> : <Volume2 />}</Button>
      </div>
      {error && <p role="status" className="px-2 text-xs text-destructive">{error}</p>}
    </div>
  </div>;
}
