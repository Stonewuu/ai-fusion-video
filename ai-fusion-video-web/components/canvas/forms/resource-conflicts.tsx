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
import {
  conflictingFields,
  type CanvasResourceDraft,
} from "@/lib/canvas/resource-draft";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";
import type { CanvasResource } from "@/lib/canvas/types";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { CanvasContentValue } from "../shared/canvas-content-value";
import { CanvasGuidance } from "../shared/canvas-guidance";

export function CanvasResourceConflicts({
  draft,
  current,
  onRebase,
  disabled,
}: {
  draft: CanvasResourceDraft;
  current: CanvasResource;
  onRebase: (serverFields: string[]) => void;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const versionKey = `${current.header.resourceType}:${current.header.resourceId}:${current.header.version}:${current.data.assetVersion}`;
  const [selection, setSelection] = useState<{
    key: string;
    choices: Record<string, string>;
  }>({ key: "", choices: {} });
  const choices = selection.key === versionKey ? selection.choices : {};
  const conflicts = conflictingFields(draft, current);
  const parentChanged =
    draft.baseline.data.assetVersion !== current.data.assetVersion;
  const changed =
    current.header.version > draft.baseline.header.version || parentChanged;
  if (!changed) return null;
  const labels = Object.fromEntries(
    RESOURCE_FIELDS[current.header.resourceType].map((field) => [
      field.key,
      field.label,
    ]),
  );
  return (
    <CanvasGuidance title="有新的修改需要核对" tone="warning">
      <p className="text-sm">
        其他位置已更新这份内容，你的草稿仍保留。
        {parentChanged ? "关联的主资产也有变化。" : ""}请比较后选择要保留的内容。
      </p>
      <Button
        variant="outline"
        size="sm"
        disabled={disabled}
        onClick={() => {
          setSelection({ key: versionKey, choices: {} });
          setOpen(true);
        }}
      >
        比较并合并
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="flex max-h-[calc(100vh-2rem)] sm:max-w-3xl flex-col overflow-hidden">
          <DialogHeader className="shrink-0">
            <DialogTitle>合并草稿</DialogTitle>
            <DialogDescription>
              逐项选择冲突字段；其他草稿字段保留，保存时仍会检查最新版本。
            </DialogDescription>
          </DialogHeader>
          <DialogScrollBody className="space-y-4">
            {conflicts.map((key) => (
              <section
                key={key}
                className="space-y-2 border-b border-border/20 pb-4"
              >
                <h4 className="text-sm font-medium">{labels[key] ?? key}</h4>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="flex flex-col gap-2 rounded-lg border border-border/30 bg-background/70 p-3">
                    <p className="text-xs text-muted-foreground">当前内容</p>
                    <CanvasContentValue type={current.header.resourceType} field={key} value={current.data[key]} />
                  </div>
                  <div className="flex flex-col gap-2 rounded-lg border border-primary/20 bg-primary/5 p-3">
                    <p className="text-xs font-medium text-sky-700 dark:text-sky-300">我的草稿</p>
                    <CanvasContentValue type={current.header.resourceType} field={key} value={draft.patch[key]} />
                  </div>
                </div>
                <CanvasSegmentedControl
                  value={choices[key] ?? ""}
                  label={`如何合并${labels[key] ?? key}`}
                  options={[
                    { value: "server", label: "采用当前", disabled },
                    { value: "mine", label: "保留草稿", disabled },
                  ]}
                  onChange={(value) =>
                    setSelection({
                      key: versionKey,
                      choices: { ...choices, [key]: value },
                    })
                  }
                />
              </section>
            ))}
            {!conflicts.length && (
              <p className="text-sm text-muted-foreground">
                草稿与最新内容没有同字段冲突。
              </p>
            )}
          </DialogScrollBody>
          <DialogFooter className="shrink-0">
            <Button variant="outline" onClick={() => setOpen(false)}>
              取消
            </Button>
            <Button
              disabled={disabled || conflicts.some((key) => !choices[key])}
              onClick={() => {
                onRebase(conflicts.filter((key) => choices[key] === "server"));
                setOpen(false);
              }}
            >
              应用合并
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </CanvasGuidance>
  );
}
