"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useState } from "react";
import { BookOpen, Copy, Trash2, FilePenLine, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { CHILD_TYPES } from "@/lib/canvas/resource-fields";
import { resourceKey, resourceTitle } from "@/lib/canvas/resource-presentation";
import { canvasTypeNames } from "@/lib/canvas/labels";
import type { CanvasResource } from "@/lib/canvas/types";
import { useCanvas } from "../canvas-provider";
import { CanvasCreateResourceDialog } from "../forms/create-resource-dialog";
import { useCanvasCollection } from "../forms/use-collection";
import { useCanvasBusinessCommand } from "../forms/use-business-command";
import { CanvasBusinessCommandState } from "../forms/business-command-state";
import { CanvasResourceExplorer } from "./resource-explorer";

export function CanvasResourceActions({
  resource,
}: {
  resource: CanvasResource;
}) {
  const [action, setAction] = useState<
    "explore" | "create" | "clone" | "delete" | "source" | null
  >(null);
  const editable = useCanvas(
    (state) =>
      state.snapshot!.permissions.canEdit &&
      !state.snapshot!.document.archived &&
      !state.pending,
  );
  const type = resource.header.resourceType;
  const child = CHILD_TYPES[type];
  const independent = [
    "asset",
    "assetItem",
    "scriptEpisode",
    "scriptScene",
    "storyboardEpisode",
    "storyboardScene",
    "shot",
  ].includes(type);
  const disabled = !editable || !resource.canEdit;
  return (
    <section
      className="space-y-3 border-t border-border/20 p-4"
      aria-label="内容结构操作"
    >
      <h3 className="text-sm font-semibold">内容与结构</h3>
      <div className="flex flex-wrap gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAction("explore")}
        >
          <BookOpen />
          {child ? "展开结构" : "阅读编辑"}
        </Button>
        {child && (
          <Button
            variant="outline"
            size="sm"
            disabled={disabled}
            onClick={() => setAction("create")}
          >
            <Plus />
            新建{canvasTypeNames[child.type]}
          </Button>
        )}
        {independent && (
          <Button
            variant="ghost"
            size="sm"
            disabled={disabled}
            onClick={() => setAction("clone")}
          >
            <Copy />
            独立副本
          </Button>
        )}
        {type === "script" && (
          <Button
            variant="ghost"
            size="sm"
            disabled={disabled}
            onClick={() => setAction("source")}
          >
            <FilePenLine />
            替换原文与结构
          </Button>
        )}
        {independent && (
          <Button
            variant="destructive-ghost"
            size="sm"
            disabled={disabled}
            onClick={() => setAction("delete")}
          >
            <Trash2 />
            删除项目内容
          </Button>
        )}
      </div>
      {action === "explore" && (
        <CanvasResourceExplorer
          resource={resource}
          onClose={() => setAction(null)}
        />
      )}
      {action === "create" && child && (
        <CanvasCreateResourceDialog
          parent={resource}
          initialType={child.type}
          onClose={() => setAction(null)}
        />
      )}
      {action === "clone" && (
        <CanvasCreateResourceDialog
          clone={resource}
          onClose={() => setAction(null)}
        />
      )}
      {(action === "source" || action === "delete") && (
        <ResourceImpactDialog
          resource={resource}
          mode={action}
          onClose={() => setAction(null)}
        />
      )}
    </section>
  );
}

function ResourceImpactDialog({
  resource,
  mode,
  onClose,
}: {
  resource: CanvasResource;
  mode: "source" | "delete";
  onClose: () => void;
}) {
  const ref = {
    type: resource.header.resourceType,
    id: resource.header.resourceId,
    version: resource.header.version,
  };
  const collection = useCanvasCollection(ref, 3);
  const [content, setContent] = useState(
    String(resource.data.rawContent ?? ""),
  );
  const [confirmedVersion, setConfirmedVersion] = useState<string | null>(null);
  const command = useCanvasBusinessCommand(
    `${mode}:${resourceKey(resource)}`,
    onClose,
  );
  const value = collection.result;
  const versionKey = value
    ? `${value.root.header.version}:${String(value.root.data.assetVersion ?? "")}`
    : null;
  const confirmed = !!versionKey && confirmedVersion === versionKey;
  const disabled =
    !value ||
    collection.loading ||
    !value.root.canEdit ||
    command.busy ||
    !!command.pending ||
    command.invalid;
  const submit = () => {
    if (!value || disabled || !confirmed) return;
    const root = value.root;
    if (mode === "source")
      command.run("script-source", {
        resourceId: ref.id,
        expectedVersion: root.header.version,
        rawContent: content,
      });
    else
      command.run("delete", {
        resourceType: ref.type,
        resourceId: ref.id,
        expectedVersion: root.header.version,
        ...(ref.type === "assetItem"
          ? { expectedParentVersion: root.data.assetVersion }
          : {}),
      });
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !command.busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] sm:max-w-2xl flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {mode === "source" ? "替换原文并清空结构" : "从项目中删除"}
          </DialogTitle>
          <DialogDescription>
            {mode === "source"
              ? "保存新原文，清空原有分集、场次以及解析摘要、人物表。之后可重新解析。"
              : "所列对象会从项目中删除，其他页面和引用画布也会受到影响。"}
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-4">
          <p className="text-sm font-medium">{resourceTitle(resource)}</p>
          {mode === "source" && (
            <Textarea
              aria-label="替换后的剧本原文"
              value={content}
              disabled={disabled}
              onChange={(event) => setContent(event.target.value)}
              className="min-h-48"
              maxLength={2000000}
            />
          )}
          {value && (
            <div className="space-y-2">
              <p className="text-sm">
                将影响 {value.descendants.length + (mode === "delete" ? 1 : 0)}{" "}
                个项目内容；原画布引用会显示对象已删除。
              </p>
              <ul className="space-y-1 text-xs text-muted-foreground">
                {value.descendants.map((entry) => (
                  <li key={resourceKey(entry)}>
                    {canvasTypeNames[entry.header.resourceType]} ·{" "}
                    {resourceTitle(entry)}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {collection.error && (
            <p role="alert" className="text-sm text-destructive">
              {collection.error}
            </p>
          )}
          <CanvasBusinessCommandState
            command={command}
            onReload={collection.refresh}
          />
          <label className="flex items-start gap-2 text-sm">
            <Checkbox
              disabled={disabled}
              checked={confirmed}
              onCheckedChange={(value) =>
                setConfirmedVersion(value === true ? versionKey : null)
              }
            />
            我已核对影响范围；画布的“撤销编辑”不会恢复被删除的项目内容。
          </label>
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={command.busy} onClick={onClose}>
            取消
          </Button>
          <Button
            variant="destructive"
            disabled={
              disabled || !confirmed || (mode === "source" && !content.trim())
            }
            onClick={submit}
          >
            {mode === "source" ? "替换并清空结构" : "删除所列对象"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
