"use client";
import { useState } from "react";
import { Eye, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { ManagedVideo } from "@/components/media/managed-video";
import { ManagedAudio } from "@/components/media/managed-audio";
import { getMediaResource } from "@/lib/api/media";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasRunValue } from "@/lib/api/canvas-runs";
import type { CanvasNode, CanvasOperation } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import { mediaNode, resourceNode } from "@/lib/canvas/placement";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasMediaPreviewDialog } from "../preview/media-preview";

export function CanvasRunResult({
  value,
  title,
  sourceNodeId,
  onSelect,
  disabled = false,
}: {
  value: CanvasRunValue;
  title: string;
  sourceNodeId: string;
  onSelect?: () => void;
  disabled?: boolean;
}) {
  const store = useCanvasStoreApi();
  const editable = useCanvas(
    (state) =>
      !!state.snapshot?.permissions.canEdit &&
      !state.snapshot.document.archived &&
      !state.pending,
  );
  const [preview, setPreview] = useState<CanvasMediaPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const reference = value.mediaId ? `media://${value.mediaId}` : null;
  const add = async () => {
    const { snapshot, catalog } = store.getState();
    if (!snapshot || !catalog) return;
    setBusy(true);
    setError(null);
    try {
      const source = snapshot.graph.nodes.find(
        (node) => node.id === sourceNodeId,
      );
      const position = {
        x: (source?.layout.x ?? 40) + 360,
        y: source?.layout.y ?? 80,
      };
      let node: CanvasNode;
      if (reference)
        node = mediaNode(await getMediaResource(reference), position);
      else if (value.resource)
        node = resourceNode(catalog, value.resource, title, position);
      else if (value.type === "text")
        node = {
          id: crypto.randomUUID(),
          title: title.slice(0, 200),
          kind: "text",
          schemaVersion: 1,
          config: { content: value.text ?? "" },
          resource: null,
          configVersion: 0,
          layoutVersion: 0,
          layout: {
            ...position,
            width: 300,
            height: null,
            parentId: null,
            locked: false,
            collapsed: false,
          },
        };
      else throw new Error("此结果没有可放入画布的资源");
      const operations: CanvasOperation[] = [
        { type: "canvas.node.create", node },
      ];
      if (source)
        operations.push({
          type: "canvas.edge.create",
          edge: {
            id: crypto.randomUUID(),
            kind: "reference",
            sourceNodeId,
            targetNodeId: node.id,
            sourcePort: null,
            targetPort: null,
            inputIndex: null,
            output: null,
            version: 0,
          },
        });
      await store.getState().edit("放入运行结果", operations);
      store.getState().focus([node.id]);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const mediaType = ["image", "video", "audio", "document"].includes(value.type)
    ? (value.type as "image" | "video" | "audio" | "document")
    : null;
  return (
    <div className="space-y-3 rounded-lg border border-border/20 bg-background/70 p-3">
      {reference && value.type === "image" ? (
        <SafeImage
          src={reference}
          alt={title}
          className="max-h-48 w-full rounded-lg object-contain"
        />
      ) : reference && value.type === "video" ? (
        <ManagedVideo
          src={reference}
          controls
          playsInline
          preload="none"
          aria-label={title}
          className="max-h-48 w-full rounded-lg object-contain"
        />
      ) : reference && value.type === "audio" ? (
        <ManagedAudio src={reference} controls className="w-full" />
      ) : value.type === "text" ? (
        <p className="line-clamp-6 whitespace-pre-wrap break-words text-sm">
          {value.text || "（空文本）"}
        </p>
      ) : (
        <p className="text-sm">{title}</p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {reference && mediaType && (
          <Button
            size="default"
            variant="outline"
            onClick={() =>
              setPreview({ kind: mediaType, source: reference, title })
            }
          >
            <Eye />
            预览
          </Button>
        )}
        {onSelect ? (
          <Button size="default" disabled={disabled} onClick={onSelect}>
            使用此结果
          </Button>
        ) : (
          <Button
            size="default"
            variant="outline"
            disabled={!editable || busy}
            onClick={() => void add()}
          >
            <Plus />
            放入画布
          </Button>
        )}
      </div>
      {error && (
        <p role="alert" className="text-xs text-destructive">
          {error}
        </p>
      )}
      {preview && (
        <CanvasMediaPreviewDialog
          media={preview}
          onClose={() => setPreview(null)}
        />
      )}
    </div>
  );
}
