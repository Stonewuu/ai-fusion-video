"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { projectApi } from "@/lib/api/project";
import type { ResourceType } from "@/lib/api/resource-events";
import { canvasTypeNames } from "@/lib/canvas/labels";
import { useCanvasLibrary } from "../library/use-library";
import { normalizeResourcePatch } from "@/lib/canvas/resource-draft";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";
import type { CanvasResource } from "@/lib/canvas/types";
import { useCanvas, useCanvasStoreApi } from "../canvas-provider";
import { CanvasSelect } from "./canvas-select";
import { CanvasResourceForm } from "./resource-form";

const parents: Partial<
  Record<ResourceType, { type: ResourceType; key: string }>
> = {
  scriptEpisode: { type: "script", key: "scriptId" },
  scriptScene: { type: "scriptEpisode", key: "episodeId" },
  assetItem: { type: "asset", key: "assetId" },
  storyboardEpisode: { type: "storyboard", key: "storyboardId" },
  storyboardScene: { type: "storyboardEpisode", key: "episodeId" },
  shot: { type: "storyboardScene", key: "storyboardSceneId" },
};
export function CanvasCreateResourceDialog({
  onClose,
  initialType = "asset",
  parent,
  clone,
}: {
  onClose: () => void;
  initialType?: ResourceType;
  parent?: CanvasResource;
  clone?: CanvasResource;
}) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const store = useCanvasStoreApi();
  const pending = useCanvas((state) => !!state.pending);
  const [type, setType] = useState<ResourceType>(
    clone?.header.resourceType ?? initialType,
  );
  const [parentId, setParentId] = useState<number | null>(
    parent?.header.resourceId ??
      (clone && parents[clone.header.resourceType]
        ? Number(clone.data[parents[clone.header.resourceType]!.key]) || null
        : null),
  );
  const [fields, setFields] = useState<Record<string, unknown>>(() => {
    if (clone) {
      const values = Object.fromEntries(
        [
          ...RESOURCE_FIELDS[clone.header.resourceType].map(
            (field) => field.key,
          ),
          "properties",
        ]
          .filter(
            (key) => key !== "episodeNumber" && clone.data[key] !== undefined,
          )
          .map((key) => [key, clone.data[key]]),
      );
      const nameKey = ["asset", "assetItem"].includes(clone.header.resourceType)
        ? "name"
        : ["scriptEpisode", "storyboardEpisode"].includes(
              clone.header.resourceType,
            )
          ? "title"
          : ["scriptScene", "storyboardScene"].includes(
                clone.header.resourceType,
              )
            ? "sceneHeading"
            : "shotNumber";
      values[nameKey] =
        `${String(values[nameKey] ?? canvasTypeNames[clone.header.resourceType])} 副本`;
      return values;
    }
    return initialType === "asset"
      ? { type: "character", name: "", status: 1 }
      : {};
  });
  const {
    entries: library,
    error: libraryError,
    loading: libraryLoading,
    refresh,
  } = useCanvasLibrary();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const parentSpec = parents[type];
  const options = parentSpec
    ? library.filter((item) => item.ref.type === parentSpec.type)
    : [];
  const chosen =
    parentId !== null
      ? options.find((item) => item.ref.id === parentId)
      : options.length === 1
        ? options[0]
        : undefined;
  const defaults =
    (type === "scriptEpisode" || type === "storyboardEpisode") && chosen
      ? {
          episodeNumber:
            Math.max(
              0,
              ...library
                .filter(
                  (item) =>
                    item.ref.type === type &&
                    item.parent === `${parentSpec!.type}:${chosen.ref.id}`,
                )
                .map((item) => item.ordinal ?? 0),
            ) + 1,
        }
      : {};
  const formFields = { ...defaults, ...fields };
  const supported: ResourceType[] =
    snapshot.document.projectId === null
      ? ["asset", "assetItem"]
      : [
          "asset",
          "assetItem",
          "scriptEpisode",
          "scriptScene",
          "storyboardEpisode",
          "storyboardScene",
          "shot",
        ];
  const create = async () => {
    setError(null);
    if (parentSpec && !chosen) {
      setError("请选择这项内容所属的剧本、分镜或资产");
      return;
    }
    let normalized: Record<string, unknown>;
    try {
      normalized = normalizeResourcePatch(type, formFields);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
      return;
    }
    if (
      type === "asset" &&
      (typeof normalized.name !== "string" || !normalized.name.trim())
    ) {
      setError("请填写资产名称");
      return;
    }
    if (
      (type === "scriptEpisode" || type === "storyboardEpisode") &&
      !normalized.episodeNumber
    ) {
      setError("请填写集号");
      return;
    }
    const nodeId = crypto.randomUUID();
    const title =
      String(
        normalized.name ??
          normalized.title ??
          normalized.sceneHeading ??
          normalized.shotNumber ??
          canvasTypeNames[type],
      ).trim() || canvasTypeNames[type];
    const body = {
      ...normalized,
      ...(parentSpec && chosen
        ? { [parentSpec.key]: chosen.ref.id, parentVersion: chosen.ref.version }
        : {}),
    };
    setBusy(true);
    try {
      await store.getState().createResource({
        requestId: crypto.randomUUID(),
        nodeId,
        title: title.slice(0, 200),
        resourceType: type,
        fields: body,
        x: 64 + (snapshot.graph.nodes.length % 4) * 360,
        y: 64 + Math.floor(snapshot.graph.nodes.length / 4) * 320,
      });
      store.getState().focus([nodeId]);
      onClose();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const initialize = async () => {
    const project = library.find((item) => item.ref.type === "project");
    if (!project) return;
    setBusy(true);
    try {
      await projectApi.initializeWorkspace(project.ref.id, project.ref.version);
      refresh();
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) onClose();
      }}
    >
      <DialogContent className="flex max-h-[calc(100vh-2rem)] sm:max-w-2xl flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>
            {clone ? "创建独立对象副本" : "新建项目内容"}
          </DialogTitle>
          <DialogDescription>
            {clone
              ? "创建一份可以单独编辑的副本，不包含下级内容。"
              : "创建后会同步到项目，并放入当前画布。"}
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-4">
          <CanvasSelect
            label="内容类型"
            value={type}
            placeholder="选择类型"
            disabled={busy || pending || !!clone || !!parent}
            items={supported.map((value) => ({
              value,
              label: canvasTypeNames[value],
            }))}
            onChange={(value) => {
              setType(value as ResourceType);
              setFields(
                value === "asset"
                  ? { type: "character", name: "", status: 1 }
                  : {},
              );
              setParentId(null);
            }}
          />
          {parentSpec && (
            <CanvasSelect
              label={`所属${canvasTypeNames[parentSpec.type]}`}
              value={chosen ? String(chosen.ref.id) : null}
              placeholder={`选择${canvasTypeNames[parentSpec.type]}`}
              disabled={busy || pending || libraryLoading}
              items={options.map((item) => ({
                value: String(item.ref.id),
                label: item.title,
              }))}
              onChange={(value) => setParentId(Number(value))}
            />
          )}
          {libraryLoading && (
            <p role="status" className="text-sm text-muted-foreground">
              正在读取项目内容…
            </p>
          )}
          {!libraryLoading &&
            !libraryError &&
            parentSpec &&
            options.length === 0 && (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  尚无可用的{canvasTypeNames[parentSpec.type]}。
                </p>
                {snapshot.document.projectId &&
                  !library.some((item) => item.ref.type === "script") && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => void initialize()}
                      disabled={busy}
                    >
                      初始化项目工作区
                    </Button>
                  )}
              </div>
            )}
          <CanvasResourceForm
            type={type}
            data={formFields}
            onChange={(key, value) =>
              setFields((current) => ({ ...current, [key]: value }))
            }
            library={library}
            disabled={busy || pending}
            assetType={chosen?.assetType}
          />
          {(error || libraryError) && (
            <p role="alert" className="text-sm text-destructive">
              {error ?? libraryError}
            </p>
          )}
          {pending && (
            <p className="text-sm text-muted-foreground">
              {busy
                ? "正在创建并同步…"
                : "创建结果待确认，请返回画布重试原操作。"}
            </p>
          )}
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button
            variant="ghost"
            disabled={busy || pending}
            onClick={() => refresh()}
          >
            刷新内容
          </Button>
          <Button variant="outline" disabled={busy} onClick={onClose}>
            {pending ? "返回画布" : "取消"}
          </Button>
          <Button disabled={busy || pending} onClick={() => void create()}>
            创建并加入画布
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
