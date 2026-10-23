"use client";

import { useId, useState } from "react";
import { FileCheck2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  canvasRunApi,
  type CanvasNodeRun,
  type CanvasRun,
} from "@/lib/api/canvas-runs";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { canvasTypeNames } from "@/lib/canvas/labels";
import { useCanvas } from "../canvas-provider";
import { useCanvasRuns } from "./canvas-runs-provider";
import { CanvasWritebackValue } from "./writeback-value";

export function CanvasWritebackDialog({
  run,
  node,
}: {
  run: CanvasRun;
  node: CanvasNodeRun;
}) {
  const runtime = useCanvasRuns();
  const editable = useCanvas(
    (state) =>
      !!state.snapshot?.permissions.canEdit &&
      !state.snapshot.document.archived,
  );
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();
  const proposal = node.writeback;
  if (!proposal) return null;
  const canDecide =
    node.status === "WAITING_CONFIRMATION" &&
    run.canControl &&
    editable &&
    !run.cancelRequested;
  const decide = async (action: "APPROVE" | "SKIP") => {
    if (!runtime.scope || !canDecide || busy) return;
    setBusy(true);
    setError(null);
    try {
      await canvasRunApi.writeback(run.id, node.id, {
        expectedRevision: node.revision,
        proposalHash: proposal.hash,
        action,
        note: note.trim(),
        acceptReset: acknowledged,
        scope: runtime.scope,
      });
      runtime.reload();
      setOpen(false);
      setNote("");
      setAcknowledged(false);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      {proposal.decision && (
        <p className="text-xs text-muted-foreground">
          {proposal.decision.action === "APPROVE"
            ? `已确认写入${proposal.field.label}`
            : "已选择跳过此次写入"}
        </p>
      )}
      <Button
        size="sm"
        variant={canDecide ? "default" : "outline"}
        onClick={() => {
          setError(null);
          setOpen(true);
        }}
      >
        <FileCheck2 />
        {canDecide ? "查看并确认写入" : "查看写入记录"}
      </Button>
      <Dialog
        open={open}
        onOpenChange={(value) => {
          if (!busy) setOpen(value);
        }}
      >
        <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-3xl">
          <DialogHeader className="shrink-0">
            <DialogTitle>
              {canDecide ? "确认写入项目内容" : "内容更新记录"}
            </DialogTitle>
            <DialogDescription>
              {canvasTypeNames[proposal.target.type]} · {proposal.targetTitle} ·
              {proposal.field.label}
            </DialogDescription>
          </DialogHeader>
          <DialogScrollBody className="space-y-4">
            {proposal.decision && (
              <p className="text-sm font-medium text-primary">
                {proposal.decision.action === "APPROVE"
                  ? "已完成写入"
                  : "已跳过，本次没有修改项目内容"}
              </p>
            )}
            {proposal.notice && (
              <p
                className={
                  proposal.dangerous
                    ? "rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive"
                    : "text-sm text-muted-foreground"
                }
              >
                {proposal.notice}
              </p>
            )}
            {proposal.removedResources && (
              <details
                className="space-y-2"
                open={proposal.removedResources.length <= 5}
              >
                <summary className="cursor-pointer text-sm font-medium">
                  {proposal.decision?.action === "APPROVE"
                    ? "已移除"
                    : proposal.decision
                      ? "原拟移除"
                      : "将移除"}{" "}
                  {
                    proposal.removedResources.filter(
                      (item) => item.type === "scriptEpisode",
                    ).length
                  }{" "}
                  个分集、
                  {
                    proposal.removedResources.filter(
                      (item) => item.type === "scriptScene",
                    ).length
                  }{" "}
                  个场次
                </summary>
                <ul className="max-h-52 overflow-auto space-y-1 text-sm text-muted-foreground">
                  {proposal.removedResources.map((item) => (
                    <li key={`${item.type}:${item.id}`}>
                      {canvasTypeNames[item.type]} ·{" "}
                      {item.title || "未命名内容"}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <div className="grid gap-4 sm:grid-cols-2">
              <section className="space-y-2">
                <h3 className="text-sm font-medium">
                  原内容
                </h3>
                <CanvasWritebackValue
                  kind={proposal.field.valueType}
                  value={proposal.before}
                  title="原内容"
                />
              </section>
              <section className="space-y-2">
                <h3 className="text-sm font-medium">
                  {proposal.decision?.action === "APPROVE"
                    ? "已写入的内容"
                    : "将写入的内容"}
                </h3>
                <CanvasWritebackValue
                  kind={proposal.field.valueType}
                  value={
                    proposal.value.type === "text"
                      ? proposal.value.text
                      : `media://${proposal.value.mediaId}`
                  }
                  title="新内容"
                />
              </section>
            </div>
            {proposal.decision ? (
              <p className="text-sm text-muted-foreground">
                处理说明：{proposal.decision.note}
              </p>
            ) : (
              canDecide && (
                <>
                  {proposal.dangerous && (
                    <div className="flex items-start gap-2">
                      <Checkbox
                        id={`${id}-ack`}
                        checked={acknowledged}
                        disabled={busy}
                        onCheckedChange={(value) =>
                          setAcknowledged(value === true)
                        }
                      />
                      <Label
                        htmlFor={`${id}-ack`}
                        className="text-sm leading-relaxed"
                      >
                        我已检查将移除的分集与场次，确认重置现有剧本解析内容。
                      </Label>
                    </div>
                  )}
                  <div className="flex flex-col gap-2">
                    <Label htmlFor={`${id}-note`}>处理说明</Label>
                    <Textarea
                      id={`${id}-note`}
                      value={note}
                      maxLength={1000}
                      rows={3}
                      disabled={busy}
                      onChange={(event) => setNote(event.target.value)}
                      placeholder="记录采用或跳过这次写入的原因"
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    如果内容已被其他人修改，会先提示你核对，保留原内容。
                  </p>
                </>
              )
            )}
            {error && (
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
            )}
          </DialogScrollBody>
          <DialogFooter className="shrink-0">
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setOpen(false)}
            >
              关闭
            </Button>
            {canDecide && (
              <>
                <Button
                  variant="secondary"
                  disabled={busy || !note.trim()}
                  onClick={() => void decide("SKIP")}
                >
                  跳过写入
                </Button>
                <Button
                  variant={proposal.dangerous ? "destructive" : "default"}
                  disabled={
                    busy ||
                    !note.trim() ||
                    (!!proposal.dangerous && !acknowledged)
                  }
                  onClick={() => void decide("APPROVE")}
                >
                  {busy
                    ? "处理中…"
                    : proposal.dangerous
                      ? "确认替换并重置"
                      : "确认写入"}
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
