"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, ListTree, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogFooter } from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { useCommandRequestKey } from "@/hooks/use-command-request-key";
import {
  scriptApi,
  type Script,
  type ScriptEpisode,
  type ScriptSourcePreview,
  type ScriptSplitRule,
} from "@/lib/api/script";
import { SourceInput } from "./source-input";
import { SourcePreview } from "./source-preview";
import { ImportStepHeader } from "./import-step-header";
import { ImportDestination } from "./import-destination";
import { cn } from "@/lib/utils";
import { MAX_EPISODE_LENGTH, deleteSegment } from "@/lib/scripts/source-import";

type PreviewDraft = ScriptSourcePreview & { source: string };
interface DeletedPreview {
  preview: PreviewDraft;
  selected: number;
  title: string;
}

interface Props {
  open: boolean;
  scriptId: number;
  scriptVersion: number;
  hasLegacySource?: boolean;
  onClose: () => void;
  onImported: (script: Script) => void;
}

export function ScriptImportDialog(props: Props) {
  return props.open ? <ImportForm {...props} /> : null;
}

function ImportForm({
  scriptId,
  scriptVersion,
  hasLegacySource = false,
  onClose,
  onImported,
}: Props) {
  const { confirm } = useConfirm();
  const requests = useCommandRequestKey();
  const [source, setSource] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [encoding, setEncoding] = useState("utf-8");
  const [rule, setRule] = useState<ScriptSplitRule>("AUTO");
  const [targetLength, setTargetLength] = useState(12000);
  const [preview, setPreview] = useState<PreviewDraft | null>(null);
  const [step, setStep] = useState<"source" | "preview">("source");
  const [previewInput, setPreviewInput] = useState<{
    source: string;
    rule: ScriptSplitRule;
    targetLength: number;
  } | null>(null);
  const [deletedPreview, setDeletedPreview] = useState<DeletedPreview | null>(
    null,
  );
  const [selected, setSelected] = useState(0);
  const [mode, setMode] = useState<"APPEND" | "REPLACE">("APPEND");
  const [version, setVersion] = useState(scriptVersion);
  const [episodes, setEpisodes] = useState<ScriptEpisode[]>([]);
  const [busy, setBusy] = useState(false);
  const [loadingExisting, setLoadingExisting] = useState(true);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState(false);
  const [submitError, setSubmitError] = useState(false);
  const hasSource = useMemo(() => source.trim().length > 0, [source]);
  const isPreview = step === "preview" && preview !== null;
  const canReusePreview =
    previewInput?.source === source &&
    previewInput.rule === rule &&
    previewInput.targetLength === targetLength;
  const existingTitles = useMemo(
    () => (mode === "APPEND" ? episodes.map((episode) => episode.title) : []),
    [episodes, mode],
  );

  useEffect(() => {
    let active = true;
    scriptApi
      .listEpisodes(scriptId)
      .then((result) => {
        if (active) {
          setEpisodes(result);
        }
      })
      .catch(() => {
        if (active) {
          setLoadError(true);
          setError("分集目录加载失败，请刷新后重试");
        }
      })
      .finally(() => {
        if (active) {
          setLoadingExisting(false);
        }
      });
    return () => {
      active = false;
    };
  }, [scriptId]);

  const refresh = async () => {
    setBusy(true);
    try {
      const [script, current] = await Promise.all([
        scriptApi.get(scriptId),
        scriptApi.listEpisodes(scriptId),
      ]);
      setVersion(script.version);
      setEpisodes(current);
      setLoadError(false);
      setSubmitError(false);
      setError("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "分集目录加载失败");
    } finally {
      setBusy(false);
    }
  };

  const loadLegacy = async () => {
    setBusy(true);
    setError("");
    try {
      const current = await scriptApi.get(scriptId);
      if (!current.rawContent?.trim()) {
        throw new Error("没有待拆分的原文，请上传文件或粘贴内容");
      }
      setFile(null);
      setSource(current.rawContent);
      setVersion(current.version);
      setMode("REPLACE");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "原文加载失败");
    } finally {
      setBusy(false);
    }
  };

  const analyze = async () => {
    if (
      !Number.isInteger(targetLength) ||
      targetLength < 1000 ||
      targetLength > MAX_EPISODE_LENGTH
    ) {
      setError("每集字数应在 1000 至 100000 之间");
      return;
    }
    if (preview && canReusePreview) {
      setStep("preview");
      setError("");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const result = await scriptApi.previewSource(
        scriptId,
        source,
        rule,
        targetLength,
      );
      setPreview({ ...result, source });
      setPreviewInput({ source, rule, targetLength });
      setStep("preview");
      setDeletedPreview(null);
      setSelected(0);
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "分集识别失败，请重试",
      );
    } finally {
      setBusy(false);
    }
  };

  const submit = async () => {
    if (!preview?.segments.length || busy) {
      return;
    }
    if (mode === "REPLACE" && episodes.length > 0) {
      const accepted = await confirm({
        title: "替换全部分集",
        variant: "destructive",
        confirmText: "替换并导入",
        description: `现有 ${episodes.length} 集及其场次将被替换。已有分镜、图片和视频保留，之后需在分镜目录重新关联剧本集。`,
      });
      if (!accepted) {
        return;
      }
    }
    setBusy(true);
    setError("");
    try {
      const payload = {
        version,
        rawContent: preview.source,
        mode,
        segments: preview.segments,
      };
      const result = await scriptApi.importSource(
        scriptId,
        payload,
        requests.forPayload(payload),
      );
      requests.clear();
      onImported(result);
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "导入失败，请重试");
      setSubmitError(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !busy) {
          onClose();
        }
      }}
    >
      <DialogContent
        showCloseButton={!busy}
        style={{ backgroundColor: "var(--background)" }}
        className={cn(
          "flex w-[calc(100%-2rem)] max-h-[calc(100dvh-2rem)] flex-col gap-0 overflow-hidden rounded-2xl p-0 sm:max-w-6xl",
          isPreview && "h-[min(48rem,calc(100dvh-2rem))]",
        )}
      >
        <ImportStepHeader preview={isPreview} />
        <DialogScrollBody
          key={step}
          className={cn(
            "flex flex-1 flex-col motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
            isPreview ? "p-0 md:overflow-hidden" : "p-4 sm:p-6",
          )}
        >
          {isPreview ? (
            <SourcePreview
              source={preview.source}
              segments={preview.segments}
              selected={selected}
              onSelect={setSelected}
              disabled={busy}
              existingTitles={existingTitles}
              onChange={(segments) => {
                setPreview({ ...preview, segments });
                setDeletedPreview(null);
              }}
              onDelete={() => {
                const result = deleteSegment(
                  preview.source,
                  preview.segments,
                  selected,
                );
                setDeletedPreview({
                  preview,
                  selected,
                  title: preview.segments[selected].title || "未填写标题",
                });
                setPreview({
                  ...preview,
                  ...result,
                  totalChars: result.source.length,
                });
                setSelected(
                  Math.min(selected, Math.max(0, result.segments.length - 1)),
                );
                setError("");
              }}
              deletedTitle={deletedPreview?.title}
              onUndoDelete={
                deletedPreview
                  ? () => {
                      setPreview(deletedPreview.preview);
                      setSelected(deletedPreview.selected);
                      setDeletedPreview(null);
                      setError("");
                    }
                  : undefined
              }
              onError={setError}
            />
          ) : (
            <SourceInput
              file={file}
              encoding={encoding}
              onFile={setFile}
              onEncoding={setEncoding}
              source={source}
              rule={rule}
              targetLength={targetLength}
              disabled={busy}
              hasLegacySource={hasLegacySource}
              onSource={setSource}
              onRule={setRule}
              onTargetLength={(length) => {
                setTargetLength(length);
                setError("");
              }}
              onError={setError}
              onLegacySource={loadLegacy}
            />
          )}
        </DialogScrollBody>
        <div className="flex shrink-0 flex-col gap-3 border-t border-border/20 bg-muted/20 px-4 py-4 sm:px-6">
          {isPreview && episodes.length > 0 && (
            <ImportDestination
              count={episodes.length}
              mode={mode}
              disabled={busy}
              onMode={setMode}
            />
          )}
          {!isPreview && preview && !canReusePreview && (
            <p className="text-xs text-muted-foreground">
              原文或规则已修改，下一步将重新生成分集。
            </p>
          )}
          {error && (
            <div className="flex shrink-0 flex-wrap items-center gap-2">
              <p role="alert" className="text-sm text-destructive">
                {error}
              </p>
              {(loadError || submitError) && (
                <Button variant="ghost" disabled={busy} onClick={refresh}>
                  刷新分集
                </Button>
              )}
            </div>
          )}
          <DialogFooter className="flex-row flex-wrap items-center justify-end gap-2">
            {isPreview && (
              <Button
                variant="ghost"
                disabled={busy}
                onClick={() => {
                  setStep("source");
                  setError("");
                }}
              >
                <ArrowLeft />
                上一步
              </Button>
            )}
            <Button
              variant="ghost"
              disabled={busy}
              onClick={onClose}
              className={isPreview ? "ml-auto" : undefined}
            >
              取消
            </Button>
            <Button
              className="max-sm:grow"
              onClick={isPreview ? submit : analyze}
              disabled={
                busy ||
                loadingExisting ||
                loadError ||
                !hasSource ||
                (isPreview &&
                  (preview.segments.length === 0 ||
                    preview.segments.some((item) => !item.title.trim()))) ||
                (!isPreview &&
                  (!Number.isInteger(targetLength) ||
                    targetLength < 1000 ||
                    targetLength > MAX_EPISODE_LENGTH))
              }
            >
              {busy ? (
                <Loader2 className="animate-spin" />
              ) : isPreview ? (
                <ListTree />
              ) : (
                <ArrowRight />
              )}
              {busy
                ? isPreview
                  ? "正在导入"
                  : "正在识别分集"
                : isPreview
                  ? `导入 ${preview.segments.length} 集`
                  : preview && canReusePreview
                    ? "继续检查分集"
                    : "识别并检查分集"}
            </Button>
          </DialogFooter>
        </div>
      </DialogContent>
    </Dialog>
  );
}
