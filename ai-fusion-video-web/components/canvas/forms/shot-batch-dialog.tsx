"use client";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { RESOURCE_FIELDS } from "@/lib/canvas/resource-fields";
import { resourceTitle } from "@/lib/canvas/resource-presentation";
import { normalizeResourcePatch } from "@/lib/canvas/resource-draft";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasResource } from "@/lib/canvas/types";
import { CanvasSelect } from "./canvas-select";
import { CanvasResourceFieldControl } from "./resource-field-control";
import { useCanvasBusinessCommand } from "./use-business-command";
import { CanvasBusinessCommandState } from "./business-command-state";

const fields = RESOURCE_FIELDS.shot.filter((field) =>
  [
    "shotType",
    "duration",
    "cameraMovement",
    "cameraAngle",
    "cameraEquipment",
    "focalLength",
    "transition",
    "soundEffect",
    "music",
    "remark",
    "status",
  ].includes(field.key),
);

export function CanvasShotBatchDialog({
  shots,
  onClose,
  onSaved,
}: {
  shots: CanvasResource[];
  onClose: () => void;
  onSaved: () => void;
}) {
  const [key, setKey] = useState(fields[0].key);
  const [value, setValue] = useState<unknown>("");
  const [error, setError] = useState<string | null>(null);
  const [reviewing, setReviewing] = useState(false);
  const field = fields.find((item) => item.key === key)!;
  const command = useCanvasBusinessCommand(
    `batch:${shots
      .map((shot) => shot.header.resourceId)
      .sort()
      .join("-")}`,
    () => {
      onSaved();
      onClose();
    },
  );
  const disabled = command.busy || !!command.pending || command.invalid;
  const save = () => {
    try {
      const board = shots[0]?.data.storyboardId;
      if (
        !board ||
        !shots.length ||
        shots.length > 500 ||
        shots.some((shot) => shot.data.storyboardId !== board || !shot.canEdit)
      )
        throw new Error("批量修改须为同一分镜中可编辑的 1 至 500 个镜头");
      const patch = normalizeResourcePatch("shot", { [key]: value });
      command.run("storyboard-batch", {
        storyboardId: board,
        entries: shots.map((shot) => ({
          resourceType: "shot",
          resourceId: shot.header.resourceId,
          expectedVersion: shot.header.version,
          patch,
        })),
      });
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    }
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
          <DialogTitle>批量修改 {shots.length} 个镜头</DialogTitle>
          <DialogDescription>
            只修改所选字段。如果有镜头被其他人修改，会先提示你核对。
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody className="space-y-4">
          <CanvasSelect
            label="修改字段"
            placeholder="选择字段"
            value={key}
            items={fields.map((item) => ({
              value: item.key,
              label: item.label,
            }))}
            disabled={disabled || reviewing}
            onChange={(next) => {
              setKey(next);
              setValue("");
            }}
          />
          <CanvasResourceFieldControl
            field={field}
            value={value}
            disabled={disabled || reviewing}
            onChange={setValue}
          />
          {reviewing && (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr>
                    <th className="p-2">镜头</th>
                    <th className="p-2">原值</th>
                    <th className="p-2">新值</th>
                  </tr>
                </thead>
                <tbody>
                  {shots.map((shot) => (
                    <tr
                      key={shot.header.resourceId}
                      className="border-t border-border/20"
                    >
                      <td className="p-2">{resourceTitle(shot)}</td>
                      <td className="p-2">{String(shot.data[key] ?? "空")}</td>
                      <td className="p-2">{String(value ?? "空")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <CanvasBusinessCommandState command={command} onReload={onSaved} />
        </DialogScrollBody>
        <DialogFooter className="shrink-0">
          <Button variant="outline" disabled={command.busy} onClick={onClose}>
            关闭
          </Button>
          {reviewing ? (
            <>
              <Button
                variant="ghost"
                disabled={disabled}
                onClick={() => setReviewing(false)}
              >
                返回修改
              </Button>
              <Button disabled={disabled} onClick={save}>
                提交全部修改
              </Button>
            </>
          ) : (
            <Button disabled={disabled} onClick={() => setReviewing(true)}>
              预览影响
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
