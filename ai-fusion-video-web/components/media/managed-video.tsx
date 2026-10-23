"use client";

import { forwardRef, type VideoHTMLAttributes } from "react";
import { useMediaUrl } from "@/hooks/use-media-url";
import { useMediaPlayback } from "@/hooks/use-media-playback";

type Props = Omit<VideoHTMLAttributes<HTMLVideoElement>, "src"> & { src?: string };
export const ManagedVideo = forwardRef<HTMLVideoElement, Props>(function ManagedVideo(
  { src, poster, ...props }, ref,
) {
  const access = useMediaUrl(src);
  const cover = useMediaUrl(poster);
  const playback = useMediaPlayback(src, access.url, props);
  if (access.error) return <span className={props.className} role="status" title={access.error}>视频不可用</span>;
  return <video {...props} {...playback} ref={ref} src={access.url ?? undefined} poster={cover.url ?? undefined} aria-busy={access.loading || undefined} />;
});
