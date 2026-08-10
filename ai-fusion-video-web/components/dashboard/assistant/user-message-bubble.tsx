"use client";

import { memo, useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import {
  File,
  FileAudio2,
  FileVideo2,
  FolderKanban,
  PlugZap,
  Sparkles,
  User,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/ui/safe-image";
import { resolveMediaUrl } from "@/lib/api/client";
import type { AgentMessage } from "@/lib/api/ai-assistant";
import {
  parseAssistantMessageReferences,
  type AssistantMessageAttachmentReference,
} from "./message-references";

export const UserMessageBubble = memo(function UserMessageBubble({
  message,
}: {
  message: AgentMessage;
}) {
  const references = parseAssistantMessageReferences(message.referencesJson);
  const hasReferenceSummary = references.projectId !== null
    || references.skills.length > 0
    || references.mcpTools.length > 0;

  return (
    <div className="flex flex-col items-end gap-1.5">
      {references.attachments.length > 0 ? (
        <div className="mr-8 flex max-w-[85%] flex-wrap justify-end gap-1.5">
          {references.attachments.map((attachment) => (
            <AttachmentThumbnail key={attachment.id} attachment={attachment} />
          ))}
        </div>
      ) : null}

      {hasReferenceSummary ? (
        <div className="mr-8 flex max-w-[85%] flex-wrap justify-end gap-1">
          {references.projectId !== null ? (
            <ReferenceBadge icon={<FolderKanban />}>
              {references.project?.name ?? `项目 #${references.projectId}`}
            </ReferenceBadge>
          ) : null}
            {references.skills.map((skill) => (
              <ReferenceBadge key={skill.id} icon={<Sparkles />}>
                Skill · {skill.displayName}
              </ReferenceBadge>
          ))}
          {references.mcpTools.map((tool) => (
            <ReferenceBadge key={`${tool.serverName}/${tool.toolName}`} icon={<PlugZap />}>
              MCP · {tool.serverName}/{tool.toolName}
            </ReferenceBadge>
          ))}
        </div>
      ) : null}

      <div className="flex w-full max-w-full justify-end gap-2">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2.5 text-sm leading-relaxed text-primary-foreground shadow-sm">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
        <span
          className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground"
          aria-hidden="true"
        >
          <User className="size-3.5" />
        </span>
      </div>
    </div>
  );
});

function ReferenceBadge({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <span className="inline-flex min-w-0 max-w-full items-center gap-1 rounded-full border border-border/30 bg-card/70 px-2 py-1 text-[10px] leading-none text-muted-foreground">
      <span className="[&>svg]:size-3" aria-hidden="true">{icon}</span>
      <span className="truncate">{children}</span>
    </span>
  );
}

function AttachmentThumbnail({ attachment }: { attachment: AssistantMessageAttachmentReference }) {
  const [previewOpen, setPreviewOpen] = useState(false);
  const source = resolveMediaUrl(attachment.resourceUrl ?? attachment.url);
  const previewable = !!source && attachment.inputType !== "file";
  const preview = attachment.inputType === "image" ? (
    <SafeImage
      src={source}
      alt={attachment.name}
      className="size-full object-cover transition-transform duration-200 ease-out group-hover:scale-[1.01] motion-reduce:transition-none"
      fallbackType="image"
    />
  ) : (
    <AttachmentIcon attachment={attachment} />
  );

  const tileClassName = "group relative flex size-14 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-border/40 bg-card text-foreground shadow-sm transition-shadow hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";
  const label = (
    <span className="pointer-events-none absolute inset-x-0 bottom-0 truncate bg-black/55 px-1 py-0.5 text-center text-[9px] leading-tight text-white backdrop-blur-sm">
      {attachment.name}
    </span>
  );

  return (
    <>
      {previewable ? (
        <button
          type="button"
          className={tileClassName}
          title={`预览 ${attachment.name}`}
          aria-label={`预览 ${attachment.name}`}
          onClick={() => setPreviewOpen(true)}
        >
          {preview}
          {label}
        </button>
      ) : source ? (
        <a
          href={source}
          target="_blank"
          rel="noreferrer"
          className={tileClassName}
          title={`打开 ${attachment.name}`}
          aria-label={`打开 ${attachment.name}`}
        >
          {preview}
          {label}
        </a>
      ) : (
        <div className={tileClassName} title={`${attachment.name}（旧消息无预览地址）`}>
          {preview}
          {label}
        </div>
      )}

      {previewOpen && source ? (
        <AttachmentPreviewDialog
          attachment={attachment}
          source={source}
          onClose={() => setPreviewOpen(false)}
        />
      ) : null}
    </>
  );
}

function AttachmentIcon({ attachment }: { attachment: AssistantMessageAttachmentReference }) {
  const icon = attachment.inputType === "video"
    ? <FileVideo2 />
    : attachment.inputType === "audio"
      ? <FileAudio2 />
      : <File />;
  return (
    <span className="flex size-8 items-center justify-center rounded-lg bg-muted text-muted-foreground [&>svg]:size-4" aria-hidden="true">
      {icon}
    </span>
  );
}

function AttachmentPreviewDialog({
  attachment,
  source,
  onClose,
}: {
  attachment: AssistantMessageAttachmentReference;
  source: string;
  onClose: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`预览 ${attachment.name}`}
      className="fixed inset-0 z-200 flex items-center justify-center bg-black/80 px-4 py-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative flex max-h-full max-w-full flex-col items-center gap-3"
        onClick={(event) => event.stopPropagation()}
      >
        <Button
          type="button"
          variant="secondary"
          size="icon-sm"
          className="absolute right-0 top-0 z-10 m-2"
          aria-label="关闭预览"
          onClick={onClose}
        >
          <X />
        </Button>

        {attachment.inputType === "image" ? (
          <SafeImage
            src={source}
            alt={attachment.name}
            className="max-h-[calc(100vh-7rem)] max-w-[calc(100vw-2rem)] rounded-xl object-contain shadow-2xl"
            fallbackType="image"
          />
        ) : attachment.inputType === "video" ? (
          <video
            src={source}
            controls
            className="max-h-[calc(100vh-7rem)] max-w-[calc(100vw-2rem)] rounded-xl bg-black shadow-2xl"
            aria-label={attachment.name}
          />
        ) : (
          <div className="w-[min(32rem,calc(100vw-2rem))] rounded-2xl border border-border/40 bg-popover/90 p-4 shadow-2xl backdrop-blur-xl">
            <p className="mb-3 truncate pr-10 text-sm text-foreground">{attachment.name}</p>
            <audio src={source} controls className="w-full" aria-label={attachment.name} />
          </div>
        )}

        <span className="max-w-[calc(100vw-2rem)] truncate rounded-full bg-black/60 px-3 py-1 text-xs text-white">
          {attachment.name} · {formatFileSize(attachment.size)}
        </span>
      </div>
    </div>,
    document.body,
  );
}

function formatFileSize(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(size < 10 * 1024 ? 1 : 0)} KB`;
  return `${(size / (1024 * 1024)).toFixed(size < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}
