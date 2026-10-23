"use client";

import { SafeImage as Image, type SafeImageProps } from "@/components/ui/safe-image";
import { useMediaUrl } from "@/hooks/use-media-url";

export type { SafeImageProps };
export function SafeImage({ src, title, alt, ...props }: SafeImageProps) {
  const access = useMediaUrl(src);
  if (access.error) return <span className={props.className} role="status" title={access.error}>图片不可用</span>;
  return <Image {...props} src={access.url} alt={alt ?? ""} title={title} aria-busy={access.loading || undefined} />;
}
