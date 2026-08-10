"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent,
} from "react";
import type { AiModel } from "@/lib/api/ai-model";
import {
  attachmentCompatibilityError,
  multimodalCapabilitySummary,
  prepareAssistantAttachments,
  type AssistantAttachment,
} from "./assistant-multimodal";

interface UseAssistantAttachmentsOptions {
  conversationId: string | null;
  model: AiModel | null;
  onError: (error: unknown) => void;
}

export function useAssistantAttachments({
  conversationId,
  model,
  onError,
}: UseAssistantAttachmentsOptions) {
  const [attachments, setAttachments] = useState<AssistantAttachment[]>([]);
  const [uploading, setUploading] = useState(false);
  const attachmentsRef = useRef<AssistantAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scopeVersionRef = useRef(0);
  const preparingRef = useRef(false);

  useEffect(() => {
    attachmentsRef.current = attachments;
  }, [attachments]);

  useEffect(() => {
    scopeVersionRef.current += 1;
    attachmentsRef.current = [];
    setAttachments([]);
  }, [conversationId]);

  const addFiles = useCallback(async (files: File[]) => {
    if (!files.length) return;
    if (!model) throw new Error("请先选择对话模型");
    if (preparingRef.current) throw new Error("附件正在处理中，请稍候");
    const scopeVersion = scopeVersionRef.current;
    preparingRef.current = true;
    setUploading(true);
    try {
      const next = await prepareAssistantAttachments(files, model, attachmentsRef.current);
      if (scopeVersion !== scopeVersionRef.current) {
        return;
      }
      setAttachments((current) => {
        const merged = [...current, ...next];
        attachmentsRef.current = merged;
        return merged;
      });
    } finally {
      preparingRef.current = false;
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }, [model]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments((current) => {
      const next = current.filter((item) => item.id !== id);
      attachmentsRef.current = next;
      return next;
    });
  }, []);

  const clearAttachments = useCallback(() => {
    setAttachments(() => {
      attachmentsRef.current = [];
      return [];
    });
  }, []);

  const handlePaste = useCallback((event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter((file): file is File => !!file);
    if (!files.length) return;
    event.preventDefault();
    void addFiles(files).catch(onError);
  }, [addFiles, onError]);

  const accept = useMemo(() => {
    const types = model?.multimodalInputTypes ?? [];
    return [
      types.includes("image") ? "image/*" : null,
      types.includes("video") ? "video/*" : null,
      types.includes("audio") ? "audio/*" : null,
      types.includes("file") ? ".pdf,.txt,.md,.csv,.json" : null,
    ].filter(Boolean).join(",");
  }, [model]);

  return {
    attachments,
    uploading,
    fileInputRef,
    accept,
    compatibilityError: attachmentCompatibilityError(model, attachments),
    capabilitySummary: multimodalCapabilitySummary(model),
    canAdd: !!model?.multimodalInputTypes.length,
    addFiles,
    removeAttachment,
    clearAttachments,
    handlePaste,
  };
}
