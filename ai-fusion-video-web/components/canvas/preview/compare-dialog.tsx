"use client";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { SafeImage } from "@/components/media/managed-image";
import { ManagedVideo } from "@/components/media/managed-video";
import { ManagedAudio } from "@/components/media/managed-audio";
import { nodeMedia, nodeSummary } from "@/lib/canvas/presentation";
import { canvasFieldLabels } from "@/lib/canvas/labels";
import type { CanvasNode } from "@/lib/canvas/types";
import { useCanvas, useCanvasResource } from "../canvas-provider";

export function CanvasCompareDialog({
  ids,
  onClose,
}: {
  ids: string[];
  onClose: () => void;
}) {
  const allNodes = useCanvas((state) => state.snapshot!.graph.nodes);
  const nodes = allNodes.filter((node) => ids.includes(node.id));
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-6xl">
        <DialogHeader className="shrink-0">
          <DialogTitle>并排对比</DialogTitle>
          <DialogDescription>
            比较所选节点的内容、素材和已保存参数。
          </DialogDescription>
        </DialogHeader>
        <div className="grid min-h-0 gap-4 overflow-y-auto md:grid-cols-2">
          {nodes.map((node) => (
            <CompareItem key={node.id} node={node} />
          ))}
        </div>
        <div className="flex shrink-0 justify-end">
          <Button variant="outline" onClick={onClose}>
            关闭
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
function CompareItem({ node }: { node: CanvasNode }) {
  const resource = useCanvasResource(node.resource);
  const media = nodeMedia(node, resource.data);
  return (
    <section className="space-y-3 rounded-xl border border-border/30 bg-card p-4">
      <h3 className="text-sm font-semibold">{node.title}</h3>
      {media?.kind === "image" && (
        <SafeImage
          src={media.source}
          alt={node.title}
          className="max-h-80 w-full object-contain"
        />
      )}
      {media?.kind === "video" && (
        <ManagedVideo
          src={media.source}
          controls
          playsInline
          preload="metadata"
          className="max-h-80 w-full"
        />
      )}
      {media?.kind === "audio" && (
        <ManagedAudio
          src={media.source}
          controls
          preload="metadata"
          className="w-full"
        />
      )}
      {resource.error && (
        <p role="alert" className="text-xs text-destructive">
          {resource.error}
        </p>
      )}
      <p className="whitespace-pre-wrap break-words text-sm">
        {nodeSummary(node, resource.data)}
      </p>
      <dl className="space-y-2 text-xs">
        {Object.entries(node.config)
          .filter(([key]) => !["resourceId", "content"].includes(key))
          .map(([key, value]) => (
            <div key={key}>
              <dt className="text-muted-foreground">
                {canvasFieldLabels[key] ?? key}
              </dt>
              <dd className="whitespace-pre-wrap break-words">
                {String(value)}
              </dd>
            </div>
          ))}
      </dl>
    </section>
  );
}
