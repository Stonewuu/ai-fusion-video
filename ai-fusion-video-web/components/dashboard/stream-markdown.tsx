"use client";

import { memo, type CSSProperties } from "react";
import { XMarkdown } from "@ant-design/x-markdown";
import { useTheme } from "next-themes";
import { cn } from "@/lib/utils";

interface StreamMarkdownProps {
  content: string;
  streaming?: boolean;
  compact?: boolean;
  tone?: "default" | "muted";
  className?: string;
}

export const StreamMarkdown = memo(function StreamMarkdown({
  content,
  streaming = false,
  compact = false,
  tone = "default",
  className,
}: StreamMarkdownProps) {
  const { resolvedTheme } = useTheme();
  const markdownThemeClass =
    resolvedTheme === "light" ? "x-markdown-light" : "x-markdown-dark";

  const colorVars =
    tone === "muted"
      ? {
          "--heading-color": "var(--foreground)",
          "--text-color": "var(--muted-foreground)",
          "--xmd-tail-color": "var(--muted-foreground)",
        }
      : {
          "--heading-color": "var(--foreground)",
          "--text-color": "var(--foreground)",
          "--xmd-tail-color": "var(--foreground)",
        };

  const markdownStyle = {
    "--font-size": compact ? "11px" : "12px",
    "--code-inline-text": compact ? "0.82em" : "0.84em",
    "--primary-color": "var(--primary)",
    "--primary-color-hover": "var(--primary)",
    "--border-color": "var(--border)",
    "--line-color": "var(--border)",
    "--light-bg": "var(--muted)",
    "--dark-bg": "var(--muted)",
    "--table-head-bg": "var(--muted)",
    "--table-body-bg": "var(--card)",
    "--cite-bg": "var(--muted)",
    "--cite-hover-bg": "var(--accent)",
    ...colorVars,
    lineHeight: compact ? 1.6 : 1.65,
  } as CSSProperties;

  return (
    <XMarkdown
      content={content}
      rootClassName={markdownThemeClass}
      streaming={
        streaming
          ? { hasNextChunk: true, tail: true, enableAnimation: true }
          : undefined
      }
      style={markdownStyle}
      className={cn(
        "min-w-0 break-words",
        compact
          ? "[&_ol]:my-1.5 [&_p]:my-1.5 [&_pre]:my-1.5 [&_ul]:my-1.5"
          : "[&_ol]:my-2 [&_p]:my-2 [&_pre]:my-2 [&_ul]:my-2",
        "[&_ol:first-child]:mt-0 [&_ol:last-child]:mb-0 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_pre:first-child]:mt-0 [&_pre:last-child]:mb-0 [&_ul:first-child]:mt-0 [&_ul:last-child]:mb-0",
        className
      )}
    />
  );
});
