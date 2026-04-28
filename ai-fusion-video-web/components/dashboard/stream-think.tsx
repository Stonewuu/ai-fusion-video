"use client";

import { useCallback, useEffect, useRef } from "react";
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

export function StreamThink({
  title,
  content,
  streaming = false,
  compact = false,
  maxHeight = 192,
  className,
}: StreamThinkProps) {
  const scrollRef = useRef<HTMLElement | null>(null);
  const userScrolledUpRef = useRef(false);

  const setThinkRef = useCallback(
    (instance: { nativeElement?: HTMLElement | null } | null) => {
      scrollRef.current = instance?.nativeElement ?? null;
    },
    []
  );

  const isNearBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const onScroll = () => {
      userScrolledUpRef.current = !isNearBottom();
    };

    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [isNearBottom]);

  useEffect(() => {
    if (!streaming) {
      userScrolledUpRef.current = false;
    }
  }, [streaming]);

  useEffect(() => {
    if (!streaming || userScrolledUpRef.current) return;

    const frameId = requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (!el || userScrolledUpRef.current) return;
      el.scrollTop = el.scrollHeight;
    });

    return () => cancelAnimationFrame(frameId);
  }, [content, streaming]);

  return (
    <Think
      ref={setThinkRef}
      className={className}
      style={{ maxHeight, overflowY: "auto" }}
      title={title}
    >
      <StreamMarkdown
        content={content}
        compact={compact}
        streaming={streaming}
      />
    </Think>
  );
}