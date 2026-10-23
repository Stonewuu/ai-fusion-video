"use client";

import { forwardRef, type AnchorHTMLAttributes } from "react";
import { useMediaUrl } from "@/hooks/use-media-url";

export const ManagedMediaLink = forwardRef<HTMLAnchorElement, AnchorHTMLAttributes<HTMLAnchorElement>>(function ManagedMediaLink(
  { href, download, onClick, title, ...props }, ref,
) {
  const access = useMediaUrl(href, download !== undefined && download !== false);
  return <a {...props} ref={ref} href={access.url ?? undefined} download={download}
    title={access.error ?? title} aria-disabled={!access.url || undefined} aria-busy={access.loading || undefined}
    onClick={(event) => { if (!access.url) event.preventDefault(); else onClick?.(event); }} />;
});
