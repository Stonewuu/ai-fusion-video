"use client";

import { useEffect, useState } from "react";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { Save, RotateCcw, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SafeImage } from "@/components/media/managed-image";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";
import {
  loadCanvasLibrary,
  type CanvasLibraryEntry,
} from "@/lib/canvas/resource-library";
import { canvasTypeNames } from "@/lib/canvas/labels";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasResource } from "@/lib/canvas/types";
import type { CanvasMediaPreview } from "@/lib/canvas/presentation";
import type { DialogueElement } from "@/lib/api/script";
import { useCanvas } from "../canvas-provider";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { useCanvasResourceDraft } from "./use-resource-draft";
import { CanvasResourceForm } from "./resource-form";
import { CanvasResourceConflicts } from "./resource-conflicts";

export function CanvasResourceEditor({
  resource,
  onPreview,
}: {
  resource: CanvasResource;
  onPreview: (media: CanvasMediaPreview) => void;
}) {
  const form = useCanvasResourceDraft(resource);
  const [mode, setMode] = useState<"read" | "edit">("read");
  const [library, setLibrary] = useState<CanvasLibraryEntry[]>([]);
  const [libraryError, setLibraryError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const revision = useResourceChangeStore(
    (state) =>
      `${state.revisions.projects}:${state.revisions.scripts}:${state.revisions.assets}:${state.revisions.storyboards}`,
  );
  const canvasEditable = useCanvas(
    (state) =>
      state.snapshot!.permissions.canEdit && !state.snapshot!.document.archived,
  );
  const editable = canvasEditable && resource.canEdit === true;
  useEffect(() => {
    let active = true;
    void loadCanvasLibrary(resource.header.projectId)
      .then((items) => {
        if (active) {
          setLibrary(items);
          setLibraryError(null);
        }
      })
      .catch((error) => {
        if (active) setLibraryError(getApiErrorMessage(error));
      });
    return () => {
      active = false;
    };
  }, [resource.header.projectId, attempt, revision]);
  if (!form.draft)
    return (
      <div role="alert" className="space-y-3 p-4">
        <p className="text-sm text-destructive">{form.error}</p>
        <Button variant="outline" onClick={form.reset}>
          清理无效草稿
        </Button>
      </div>
    );
  const { draft } = form;
  const dirty = Object.keys(draft.patch).length > 0;
  const data = { ...draft.baseline.data, ...draft.patch };
  const currentEntry = library.find(
    (item) =>
      item.ref.type === resource.header.resourceType &&
      item.ref.id === resource.header.resourceId,
  );
  const title = String(
    data.name ??
      data.title ??
      data.sceneHeading ??
      data.shotNumber ??
      canvasTypeNames[resource.header.resourceType],
  );
  return (
    <section
      className="space-y-4 p-4"
      aria-label={`${canvasTypeNames[resource.header.resourceType]}编辑器`}
    >
      <div className="space-y-2">
        <h3 className="break-words text-base font-semibold">{title}</h3>
        <p className="text-xs text-muted-foreground">
          {canvasTypeNames[resource.header.resourceType]}
          {dirty ? " · 有未保存草稿" : ""}
        </p>
        <CanvasSegmentedControl
          label="内容查看方式"
          value={mode}
          onChange={setMode}
          options={[
            { value: "read", label: "阅读" },
            { value: "edit", label: "编辑", disabled: !editable },
          ]}
        />
      </div>
      {(dirty || form.pending) && (
        <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b border-border/20 bg-card py-3">
          <Button
            disabled={!editable || form.saving}
            onClick={() => void form.save()}
          >
            {form.saving ? (
              <Loader2 className="animate-spin motion-reduce:animate-none" />
            ) : (
              <Save />
            )}
            {form.pending ? "再次确认" : "保存内容"}
          </Button>
          <Button
            variant="outline"
            disabled={form.saving || !!form.pending}
            onClick={form.reset}
          >
            <RotateCcw />
            放弃草稿
          </Button>
        </div>
      )}
      {dirty && (
        <CanvasResourceConflicts
          draft={draft}
          current={resource}
          onRebase={form.rebase}
          disabled={form.saving || !!form.pending}
        />
      )}
      {form.error && (
        <p role="alert" className="text-sm text-destructive">
          {form.error}
        </p>
      )}
      {form.storageError && (
        <p role="alert" className="text-sm text-destructive">
          {form.storageError}，离开前请保存。
        </p>
      )}
      {libraryError && (
        <div className="space-y-2 text-sm text-destructive">
          <p>关联对象读取失败：{libraryError}</p>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setAttempt(attempt + 1)}
          >
            重新读取
          </Button>
        </div>
      )}
      {form.pending && !form.saving && (
        <p role="status" className="text-sm text-muted-foreground">
          还未确认上次保存是否完成。点击“再次确认”可查看结果，不会重复保存。
        </p>
      )}
      {mode === "edit" ? (
        <CanvasResourceForm
          type={resource.header.resourceType}
          data={data}
          onChange={form.change}
          library={library}
          disabled={!editable}
          assetType={currentEntry?.assetType}
        />
      ) : (
        <CanvasResourceRead
          resource={{ ...resource, data }}
          library={library}
          onPreview={onPreview}
        />
      )}

    </section>
  );
}

export function CanvasResourceRead({
  resource,
  library,
  onPreview,
}: {
  resource: CanvasResource;
  library: CanvasLibraryEntry[];
  onPreview: (media: CanvasMediaPreview) => void;
}) {
  const data = resource.data;
  return (
    <div className="space-y-5">
      {RESOURCE_FIELDS[resource.header.resourceType].map((field) => {
        const value = data[field.key];
        if (
          value === null ||
          value === undefined ||
          value === "" ||
          (Array.isArray(value) && !value.length)
        )
          return null;
        let content;
        if (field.kind === "image" && typeof value === "string")
          content = (
            <div className="space-y-2">
              <SafeImage
                src={value}
                alt={field.label}
                className="max-h-64 w-full rounded-lg object-contain"
              />
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  onPreview({
                    source: value,
                    kind: "image",
                    title: field.label,
                  })
                }
              >
                预览
              </Button>
            </div>
          );
        else if (
          (field.kind === "video" || field.kind === "audio") &&
          typeof value === "string"
        )
          content = (
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                onPreview({
                  source: value,
                  kind: field.kind as "video" | "audio",
                  title: field.label,
                })
              }
            >
              预览{field.label}
            </Button>
          );
        else if (field.kind === "dialogues" && Array.isArray(value))
          content = (
            <div className="space-y-3">
              {(value as DialogueElement[]).map((item, index) => (
                <div key={index}>
                  {item.character_name && (
                    <p className="text-sm font-medium">
                      {item.character_name}
                      {item.parenthetical ? `（${item.parenthetical}）` : ""}
                    </p>
                  )}
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">
                    {item.content}
                  </p>
                </div>
              ))}
            </div>
          );
        else if (["asset", "assetItem", "scriptEpisode"].includes(field.kind))
          content = (
            <p className="text-sm">
              {(Array.isArray(value) ? value : [value])
                .map(
                  (id) =>
                    library.find(
                      (item) =>
                        item.ref.type === field.kind && item.ref.id === id,
                    )?.title ?? "关联内容暂不可读取",
                )
                .join("、")}
            </p>
          );
        else if (field.kind === "choice")
          content = (
            <p className="text-sm">
              {field.options?.find((option) => option.value === String(value))
                ?.label ?? String(value)}
            </p>
          );
        else
          content = (
            <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {typeof value === "object"
                ? JSON.stringify(value, null, 2)
                : String(value)}
            </p>
          );
        return (
          <div key={field.key} className="space-y-2">
            <h4 className="text-xs font-medium text-muted-foreground">
              {field.label}
            </h4>
            {content}
          </div>
        );
      })}
      {data.properties && typeof data.properties === "object" ? (
        <details>
          <summary className="cursor-pointer text-sm font-medium">
            资产属性
          </summary>
          <dl className="mt-3 space-y-2">
            {Object.entries(data.properties).map(([key, value]) => (
              <div key={key}>
                <dt className="text-xs text-muted-foreground">{key}</dt>
                <dd className="whitespace-pre-wrap text-sm">
                  {typeof value === "string" ? value : JSON.stringify(value)}
                </dd>
              </div>
            ))}
          </dl>
        </details>
      ) : null}
    </div>
  );
}
