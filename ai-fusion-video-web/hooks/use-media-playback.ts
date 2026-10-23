"use client";

import { useRef, type ReactEventHandler } from "react";

interface Events<T extends HTMLMediaElement> {
  onTimeUpdate?: ReactEventHandler<T>;
  onPlay?: ReactEventHandler<T>;
  onPause?: ReactEventHandler<T>;
  onLoadedMetadata?: ReactEventHandler<T>;
}

/** 换票据保留播放位置和用户暂停状态，换资源时从新资源本身的状态开始。 */
export function useMediaPlayback<T extends HTMLMediaElement>(source: string | undefined, url: string | null, events: Events<T>): Events<T> {
  const saved = useRef({ source, url, time: 0, playing: false });
  return {
    onTimeUpdate: (event) => {
      if (saved.current.source === source && saved.current.url === url && event.currentTarget.readyState > 0) {
        saved.current.time = event.currentTarget.currentTime;
      }
      events.onTimeUpdate?.(event);
    },
    onPlay: (event) => {
      saved.current = { source, url, time: event.currentTarget.currentTime, playing: true };
      events.onPlay?.(event);
    },
    onPause: (event) => {
      // 浏览器替换 src 时会发出 pause；这不代表用户要求暂停。
      if (saved.current.source === source && saved.current.url === url && event.currentTarget.readyState > 0) saved.current.playing = false;
      events.onPause?.(event);
    },
    onLoadedMetadata: (event) => {
      const media = event.currentTarget;
      const previous = saved.current;
      if (previous.source === source && previous.url !== null && previous.url !== url) {
        media.currentTime = Math.min(previous.time, Number.isFinite(media.duration) ? media.duration : previous.time);
        saved.current = { source, url, time: media.currentTime, playing: previous.playing };
        if (previous.playing) void media.play().catch(() => { saved.current.playing = false; });
        else media.pause();
      } else {
        saved.current = { source, url, time: media.currentTime, playing: !media.paused };
      }
      events.onLoadedMetadata?.(event);
    },
  };
}
