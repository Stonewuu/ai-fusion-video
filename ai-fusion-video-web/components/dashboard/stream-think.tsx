"use client";

import { memo, useCallback, useEffect, useRef, useState } from "react";
import { Think } from "@ant-design/x";
import { StreamMarkdown } from "@/components/dashboard/stream-markdown";

interface StreamThinkProps {
  title: string;
  content: string;
  streaming?: boolean;
  compact?: boolean;
  maxHeight?: number;
  className?: string;
}

export const StreamThink = memo(function StreamThink({
  title,
  content,
  streaming = false,
  compact = false,
  maxHeight = 192,
  className,
}: StreamThinkProps) {
  const rootRef = useRef<HTMLElement | null>(null);
  const contentRef = useRef<HTMLElement | null>(null);
  const previousStreamingRef = useRef(streaming);
  const [expanded, setExpanded] = useState(streaming);

  const findScrollableContent = useCallback(() => {
    const content = rootRef.current?.querySelector<HTMLElement>(
      ".stream-think-scroll-content"
    ) ?? null;
    contentRef.current = content;
    return content;
  }, []);

  const setThinkRef = useCallback(
    (instance: { nativeElement?: HTMLElement | null } | null) => {
      rootRef.current = instance?.nativeElement ?? null;
      findScrollableContent();
    },
    [findScrollableContent]
  );

  useEffect(() => {
    const shouldSyncExpanded = streaming || previousStreamingRef.current;
    previousStreamingRef.current = streaming;
    if (!shouldSyncExpanded) return;

    const frameId = requestAnimationFrame(() => setExpanded(streaming));
    return () => cancelAnimationFrame(frameId);
  }, [streaming]);

  useEffect(() => {
    if (!expanded) return;

    const frameId = requestAnimationFrame(findScrollableContent);
    return () => cancelAnimationFrame(frameId);
  }, [expanded, findScrollableContent]);

  useEffect(() => {
    if (!streaming || !expanded) return;

    const frameId = requestAnimationFrame(() => {
      const el = contentRef.current ?? findScrollableContent();
      if (!el) return;
      el.scrollTop = el.scrollHeight;
    });

    return () => cancelAnimationFrame(frameId);
  }, [content, expanded, findScrollableContent, streaming]);

  const thinkStyles = {
    status: {
      color: "var(--muted-foreground)",
    },
    content: {
      color: "var(--foreground)",
      borderInlineStartColor: "var(--border)",
      maxHeight: Math.max(0, maxHeight - 32),
      overflowY: "auto" as const,
    },
  };

  return (
    <Think
      ref={setThinkRef}
      className={className}
      style={{ maxHeight }}
      styles={thinkStyles}
      classNames={{ content: "stream-think-scroll-content" }}
      title={title}
      expanded={expanded}
      onExpand={setExpanded}
    >
      <StreamMarkdown
        content={content}
        compact={compact}
        streaming={streaming}
      />
    </Think>
  );
});
