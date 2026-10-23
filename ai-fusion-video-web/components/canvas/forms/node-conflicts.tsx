"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  nodeDraftConflicts,
  nodeDraftValue,
  rebaseNodeDraft,
  type CanvasNodeDraft,
} from "@/lib/canvas/node-draft";
import { canvasFieldLabels } from "@/lib/canvas/labels";
import type { CanvasNode } from "@/lib/canvas/types";
import { CanvasSegmentedControl } from "../workspace/segmented-control";

export function CanvasNodeConflicts({
  draft,
  current,
  onRebase,
  disabled,
}: {
  draft: CanvasNodeDraft;
  current: CanvasNode;
  onRebase: (value: CanvasNodeDraft) => void;
  disabled: boolean;
}) {
  const versionKey = `${current.id}:${current.configVersion}`;
  const [selection, setSelection] = useState<{
    key: string;
    choices: Record<string, string>;
  }>({ key: "", choices: {} });
  const choices = selection.key === versionKey ? selection.choices : {};
  const conflicts = nodeDraftConflicts(draft, current);
  return (
    <section className="space-y-3 rounded-lg border border-border/40 bg-muted/50 p-3">
      <p className="text-sm">节点已更新。请核对变化，合并后再保存。</p>
      {conflicts.map((key) => (
        <div key={key} className="space-y-2">
          <p className="text-sm font-medium">
            {key === "@title" ? "节点名称" : (canvasFieldLabels[key] ?? key)}
          </p>
          <p className="break-words whitespace-pre-wrap text-xs text-muted-foreground">
            当前内容：{String(nodeDraftValue(current, key) ?? "（未设置）")}
          </p>
          <p className="break-words whitespace-pre-wrap text-xs">
            我的草稿：{String(nodeDraftValue(draft, key) ?? "（清空）")}
          </p>
          <CanvasSegmentedControl
            label={`合并${key === "@title" ? "节点名称" : (canvasFieldLabels[key] ?? key)}`}
            value={choices[key] ?? ""}
            className="w-full"
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
        </div>
      ))}
      {!conflicts.length && (
        <p className="text-xs text-muted-foreground">
          本地修改与最新内容没有同字段冲突。
        </p>
      )}
      <Button
        size="sm"
        variant="outline"
        disabled={disabled || conflicts.some((key) => !choices[key])}
        onClick={() =>
          onRebase(
            rebaseNodeDraft(
              draft,
              current,
              conflicts.filter((key) => choices[key] === "server"),
            ),
          )
        }
      >
        合并更新
      </Button>
    </section>
  );
}
