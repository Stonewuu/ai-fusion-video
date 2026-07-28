"use client";

import { memo, type CSSProperties } from "react";
import { Actions } from "@ant-design/x";
import { XMarkdown, type ComponentProps } from "@ant-design/x-markdown";
import { useTheme } from "next-themes";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import { cn } from "@/lib/utils";

interface StreamMarkdownProps {
  content: string;
  streaming?: boolean;
  compact?: boolean;
  tone?: "default" | "muted";
  className?: string;
}

function MarkdownTable({ children }: ComponentProps) {
  return (
    <div className="my-3 w-full overflow-hidden rounded-lg border border-border/30 bg-background/70">
      <div className="w-full overflow-x-auto">
        <table className="!m-0 !table !w-full min-w-max !max-w-none !border-separate !border-spacing-0 !overflow-visible text-left [&_tbody_td]:!border-0 [&_tbody_td]:!border-t [&_tbody_td]:!border-border/30 [&_th]:!border-0 [&_tr>*+*]:!border-l [&_tr>*+*]:!border-border/30">
          {children}
        </table>
      </div>
    </div>
  );
}

function MarkdownCode({ block, children, className, lang }: ComponentProps) {
  const { resolvedTheme } = useTheme();

  if (!block) {
    return <code className={className}>{children}</code>;
  }

  if (typeof children !== "string") {
    return null;
  }

  const language =
    lang?.trim().match(/^\S+/)?.[0]?.toLowerCase() ||
    className?.match(/(?:^|\s)language-([^\s]+)/)?.[1]?.toLowerCase() ||
    "";

  return (
    <div className="overflow-hidden rounded-lg border border-border/30 bg-background/70">
      <div className="flex min-h-10 items-center justify-between bg-muted/70 px-3 py-2">
        <span className="font-mono text-xs font-semibold">
          {language || "text"}
        </span>
        <Actions.Copy text={children} aria-label="复制代码" />
      </div>
      <div className="stream-markdown-highlightCode-code">
        <SyntaxHighlighter
          language={language || "text"}
          style={resolvedTheme === "dark" ? oneDark : oneLight}
          customStyle={{ margin: 0, padding: "12px" }}
          wrapLines
        >
          {children.replace(/\n$/, "")}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

function IncompleteTable() {
  return (
    <div
      className="my-3 w-full overflow-hidden rounded-lg border border-border/30 bg-background/70"
      aria-label="表格生成中"
    >
      <div className="h-8 animate-pulse bg-muted/70 motion-reduce:animate-none" />
      <div className="grid grid-cols-2 gap-px bg-border/20">
        <span className="h-8 animate-pulse bg-card/80 motion-reduce:animate-none" />
        <span className="h-8 animate-pulse bg-card/80 motion-reduce:animate-none" />
      </div>
    </div>
  );
}

const MARKDOWN_COMPONENTS = {
  code: MarkdownCode,
  table: MarkdownTable,
  "incomplete-table": IncompleteTable,
};

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
      paragraphTag="div"
      openLinksInNewTab
      components={MARKDOWN_COMPONENTS}
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
        "[&_ol:first-child]:mt-0 [&_ol:last-child]:mb-0 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_pre:first-child]:mt-0 [&_pre:last-child]:mb-0 [&_table]:text-xs [&_th]:whitespace-nowrap [&_ul:first-child]:mt-0 [&_ul:last-child]:mb-0",
        className
      )}
    />
  );
});
