"use client";

import { forwardRef, type AudioHTMLAttributes } from "react";
import { useMediaUrl } from "@/hooks/use-media-url";
import { useMediaPlayback } from "@/hooks/use-media-playback";

type Props = Omit<AudioHTMLAttributes<HTMLAudioElement>, "src"> & { src?: string };
export const ManagedAudio = forwardRef<HTMLAudioElement, Props>(function ManagedAudio(
  { src, ...props }, ref,
) {
  const access = useMediaUrl(src);
  const playback = useMediaPlayback(src, access.url, props);
  if (access.error) return <span className={props.className} role="status" title={access.error}>音频不可用</span>;
  return <audio {...props} {...playback} ref={ref} src={access.url ?? undefined} aria-busy={access.loading || undefined} />;
});
