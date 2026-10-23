"use client";

import { useRef, useState } from "react";
import { Dialog, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { CanvasDialogContent } from "../shared/canvas-dialog";
import { CanvasMediaPicker } from "../library/media-picker";
import { CanvasNodeMedia } from "./node-media";
import { useCanvas, useCanvasResourceCache, useCanvasStoreApi } from "../canvas-provider";
import { canvasMediaKinds } from "@/lib/canvas/placement";
import { canvasApi } from "@/lib/api/canvas";
import { mediaId, type ManagedMedia } from "@/lib/api/media";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasNode, CanvasResource } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { CanvasNodeMediaGroup } from "@/lib/canvas/node-media";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";

export function CanvasNodeQuickMedia({ node, resource, groups, canEdit, onPreview }: {
  node: CanvasNode; resource: CanvasResource | null; groups: CanvasNodeMediaGroup[]; canEdit: boolean;
  onPreview: (media: CanvasMediaPreview, returnFocus?: HTMLElement) => void;
}) {
  const store = useCanvasStoreApi();
  const cache = useCanvasResourceCache();
  const pending = useCanvas((state) => !!state.pending);
  const [picking, setPicking] = useState<{ media: CanvasMediaPreview & { field?: string }; node: CanvasNode; resource: CanvasResource | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const returning = useRef<HTMLElement | null>(null);
  const request = useRef<{ mediaId: string; id: string } | null>(null);
  const ownKind = canvasMediaKinds[node.kind as keyof typeof canvasMediaKinds];
  const replace = async (media: ManagedMedia) => {
    if (!picking || busy || pending) return;
    setBusy(true); setError(null);
    try {
      const state = store.getState();
      if (!state.snapshot?.permissions.canEdit || state.snapshot.document.archived) throw new Error("当前画布不可编辑。");
      if (picking.resource && picking.media.field) {
        const baseline = picking.resource;
        if (!baseline.canEdit) throw new Error("没有修改此内容的权限。");
        if (request.current?.mediaId !== media.id) request.current = { mediaId: media.id, id: crypto.randomUUID() };
        await canvasApi.patchResource(state.canvasId, {
          requestId: request.current.id, resourceType: baseline.header.resourceType, resourceId: baseline.header.resourceId,
          expectedVersion: baseline.header.version, patch: { [picking.media.field]: `media://${media.id}` },
          ...(baseline.header.resourceType === "assetItem" ? { expectedParentVersion: baseline.data.assetVersion as number } : {}),
        });
        cache.invalidate();
      } else {
        await state.edit("更换节点素材", [{ type: "canvas.node.patch", nodeId: node.id,
          expectedConfigVersion: picking.node.configVersion, configPatch: { resourceId: media.id } }]);
      }
      setPicking(null); request.current = null;
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  return <>
    <CanvasNodeMedia groups={groups} onPreview={onPreview}
      onReplace={canEdit && !pending && (ownKind || resource?.canEdit) ? (media, trigger) => {
        returning.current = trigger.closest<HTMLElement>(".react-flow__node");
        store.getState().select([node.id]);
        request.current = null; setError(null); setPicking({ media, node, resource });
      } : undefined}
      canReplace={(media) => !!ownKind || !!resource && RESOURCE_FIELDS[resource.header.resourceType].some((field) => field.key === media.field && ["image", "video", "audio"].includes(field.kind))} />
    <Dialog open={!!picking} onOpenChange={(open) => { if (!open && !busy) setPicking(null); }}>
      <CanvasDialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden"
        finalFocus={() => returning.current}>
        <DialogHeader className="shrink-0"><DialogTitle>更换{picking?.media.kind === "video" ? "视频" : picking?.media.kind === "audio" ? "音频" : "素材"}</DialogTitle>
          <DialogDescription>选择素材即可替换，也可以上传新文件。</DialogDescription></DialogHeader>
        <DialogScrollBody>
          {error && <p role="alert" className="pb-3 text-sm text-destructive">{error}</p>}
          {picking && <CanvasMediaPicker kind={ownKind ?? (picking.media.kind === "video" ? "VIDEO" : picking.media.kind === "audio" ? "AUDIO" : "IMAGE")}
            value={mediaId(picking.media.source) ?? undefined} disabled={busy || pending}
            onSelect={(media) => void replace(media)} />}
        </DialogScrollBody>
      </CanvasDialogContent>
    </Dialog>
  </>;
}
