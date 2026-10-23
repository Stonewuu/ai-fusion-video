"use client";
import { useState } from "react";
import {
  FileText,
  Table2,
  Braces,
  Package,
  Film,
  Download,
  Check,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { cn } from "@/lib/utils";
import { getApiErrorMessage } from "@/lib/api/api-error";
import {
  canvasExportsApi,
  type CanvasExportFormat,
  type CanvasExportScope,
  type CanvasExportOptions,
  type CanvasExportPreview,
} from "@/lib/api/canvas-exports";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import { useCanvas } from "../canvas-provider";
import { CanvasSegmentedControl } from "../workspace/segmented-control";
import { CanvasExportHistory, exportSize } from "./export-history";
import { CanvasTemplateExportPanel } from "../templates/template-export-panel";

const FORMATS = [
  {
    value: "ZIP",
    label: "交付包",
    description: "文稿、分镜、媒体和来源清单",
    icon: Package,
  },
  {
    value: "MARKDOWN",
    label: "剧本文稿",
    description: "可继续编辑的 Markdown",
    icon: FileText,
  },
  {
    value: "CSV",
    label: "分镜表格",
    description: "Excel 可打开的 CSV",
    icon: Table2,
  },
  {
    value: "MEDIA",
    label: "媒体文件",
    description: "下载选中的单个媒体或成片",
    icon: Film,
  },
  {
    value: "JSON",
    label: "画布数据",
    description: "保存画布结构、内容和引用（JSON）",
    icon: Braces,
  },
] as const;
export function CanvasExportDialog({ onClose }: { onClose: () => void }) {
  const snapshot = useCanvas((state) => state.snapshot!);
  const selection = useCanvas((state) => state.selection);
  const saving = useCanvas((state) => state.saving || !!state.pending);
  const [tab, setTab] = useState<"create" | "history" | "template">("create");
  const [format, setFormat] = useState<CanvasExportFormat>("ZIP");
  const [scope, setScope] = useState<CanvasExportScope>(
    selection.length ? "SELECTION" : "CANVAS",
  );
  const [preview, setPreview] = useState<{
    result: CanvasExportPreview;
    options: CanvasExportOptions;
    key: string;
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requests = useCommandRequestKey();
  const options: CanvasExportOptions = {
    format,
    scope,
    nodeIds: scope === "SELECTION" ? selection : [],
    expectedRevision: snapshot.document.revision,
  };
  const key = JSON.stringify(options);
  const currentPreview = preview?.key === key ? preview : null;
  const inspect = async () => {
    setBusy(true);
    setError(null);
    try {
      setPreview({
        result: await canvasExportsApi.preview(snapshot.document.id, options),
        options,
        key,
      });
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setBusy(false);
    }
  };
  const submit = async () => {
    if (!currentPreview) return;
    setBusy(true);
    setError(null);
    const body = {
      options: currentPreview.options,
      previewHash: currentPreview.result.hash,
    };
    try {
      await canvasExportsApi.submit(snapshot.document.id, {
        ...body,
        requestId: requests.forPayload(body),
      });
      requests.clear();
      setPreview(null);
      setTab("history");
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
      <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
        <DialogHeader className="shrink-0">
          <DialogTitle className="flex items-center gap-2">
            <Download className="size-5 text-primary" />
            导出与交付
          </DialogTitle>
          <DialogDescription>
            {tab === "template" ? "保存创作流程和内容，用于创建新的独立项目。" : "导出文稿、分镜和媒体。开始后可关闭此窗口，稍后在导出记录中下载。"}
          </DialogDescription>
        </DialogHeader>
        <CanvasSegmentedControl
          label="交付内容"
          value={tab}
          options={[
            { value: "create", label: "导出文件", disabled: busy },
            { value: "history", label: "导出记录", disabled: busy },
            { value: "template", label: "画布模板", disabled: busy },
          ]}
          onChange={setTab}
        />
        <div className="min-h-0 overflow-y-auto px-1 py-2">
          {tab === "template" ? <CanvasTemplateExportPanel onBusyChange={setBusy} /> : tab === "history" ? (
            <CanvasExportHistory canvasId={snapshot.document.id} />
          ) : (
            <div className="flex flex-col gap-6">
              <div className="flex flex-col gap-3">
                <p className="text-sm font-medium">导出范围</p>
                <CanvasSegmentedControl
                  label="导出范围"
                  value={scope}
                  options={[
                    { value: "CANVAS", label: "当前画布", disabled: busy },
                    {
                      value: "SELECTION",
                      label: `所选节点 (${selection.length})`,
                      disabled: busy || selection.length === 0,
                    },
                    ...(snapshot.document.projectId
                      ? [
                          {
                            value: "PROJECT" as const,
                            label: "整个项目",
                            disabled: busy,
                          },
                        ]
                      : []),
                  ]}
                  onChange={setScope}
                />
              </div>
              <fieldset className="flex flex-col gap-3" disabled={busy}>
                <legend className="mb-3 text-sm font-medium">文件格式</legend>
                <div
                  role="radiogroup"
                  aria-label="文件格式"
                  className="grid gap-3 sm:grid-cols-2"
                  onKeyDown={(event) => {
                    if (
                      busy ||
                      ![
                        "ArrowRight",
                        "ArrowLeft",
                        "ArrowDown",
                        "ArrowUp",
                        "Home",
                        "End",
                      ].includes(event.key)
                    )
                      return;
                    event.preventDefault();
                    const index = FORMATS.findIndex(
                      (item) => item.value === format,
                    );
                    const next =
                      event.key === "Home"
                        ? 0
                        : event.key === "End"
                          ? FORMATS.length - 1
                          : (index +
                              (["ArrowRight", "ArrowDown"].includes(event.key)
                                ? 1
                                : FORMATS.length - 1)) %
                            FORMATS.length;
                    setFormat(FORMATS[next].value);
                    event.currentTarget
                      .querySelectorAll<HTMLButtonElement>(
                        'button[role="radio"]',
                      )
                      [next]?.focus();
                  }}
                >
                  {FORMATS.map((item) => (
                    <button
                      key={item.value}
                      type="button"
                      role="radio"
                      aria-checked={format === item.value}
                      tabIndex={format === item.value ? 0 : -1}
                      onClick={() => setFormat(item.value)}
                      className={cn(
                        "flex items-center gap-3 rounded-xl border p-4 text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
                        format === item.value
                          ? "border-primary/50 bg-primary/5 dark:border-primary/80 dark:bg-primary/20"
                          : "border-border/30 bg-card hover:bg-muted/30",
                      )}
                    >
                      <item.icon
                        className={cn(
                          "size-5 shrink-0",
                          format === item.value
                            ? "text-primary dark:text-foreground"
                            : "text-muted-foreground",
                        )}
                      />
                      <span className="min-w-0 flex-1">
                        <span className="block text-sm font-medium">
                          {item.label}
                        </span>
                        <span className="mt-1 block text-xs text-muted-foreground">
                          {item.description}
                        </span>
                      </span>
                      {format === item.value && (
                        <Check className="size-4 shrink-0 text-primary dark:text-foreground" />
                      )}
                    </button>
                  ))}
                </div>
              </fieldset>
              {currentPreview && (
                <div
                  role="status"
                  className="flex flex-col gap-3 rounded-xl border border-primary/30 bg-primary/5 p-4"
                >
                  <p className="text-sm font-medium">
                    本次交付范围
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {currentPreview.result.nodes} 个节点 ·{" "}
                    {currentPreview.result.businessObjects} 个项目内容 ·{" "}
                    {currentPreview.result.mediaFiles} 个媒体 (
                    {exportSize(currentPreview.result.mediaBytes)})
                  </p>
                  {currentPreview.result.warnings.map((warning) => (
                    <p key={warning} className="text-xs text-muted-foreground">
                      {warning}
                    </p>
                  ))}
                </div>
              )}
              <p className="text-xs leading-relaxed text-muted-foreground">
                导出会保留当前版本，之后的编辑不会影响这份文件。导出前可检查内容和素材是否齐全。
              </p>
            </div>
          )}
        </div>
        {error && (
          <p role="alert" className="shrink-0 text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter className="shrink-0">
          <Button variant="ghost" disabled={busy} onClick={onClose}>
            关闭
          </Button>
          {tab === "create" && (
            <Button
              disabled={busy || saving}
              onClick={() => void (currentPreview ? submit() : inspect())}
            >
              {busy
                ? "正在处理…"
                : currentPreview
                  ? "生成交付文件"
                  : "预览导出范围"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
