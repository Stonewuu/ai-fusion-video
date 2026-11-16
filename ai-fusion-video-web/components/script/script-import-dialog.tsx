"use client";

import { useEffect, useState } from "react";
import { ArrowLeft, ListTree, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { DialogContent } from "@/components/shared/dialog-content";
import { DialogScrollBody } from "@/components/shared/dialog-scroll-body";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
const modes = [
  { value: "APPEND", label: "追加到现有分集" },
  { value: "REPLACE", label: "替换全部分集" },
];

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
    if (targetLength < 1000 || targetLength > MAX_EPISODE_LENGTH) {
      setError("每集字数应在 1000 至 100000 之间");
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
        className="flex max-h-[calc(100dvh-2rem)] flex-col overflow-hidden sm:max-w-4xl"
      >
        <DialogHeader className="shrink-0">
          <DialogTitle>{preview ? "检查分集" : "导入原文"}</DialogTitle>
          <DialogDescription>
            {preview
              ? "确认标题和分界后，导入剧本目录。"
              : "上传或粘贴原文，按章节整理为分集。"}
          </DialogDescription>
        </DialogHeader>
        <DialogScrollBody
          className={
            preview ? "flex h-[60dvh] flex-col gap-3" : "flex flex-col gap-4"
          }
        >
          {preview ? (
            <SourcePreview
              source={preview.source}
              segments={preview.segments}
              selected={selected}
              onSelect={setSelected}
              disabled={busy}
              existingTitles={
                mode === "APPEND"
                  ? episodes.map((episode) => episode.title)
                  : []
              }
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
        {preview && episodes.length > 0 && (
          <div className="flex shrink-0 flex-wrap items-center gap-3 p-1">
            <label htmlFor="source-import-mode" className="text-sm">
              导入方式
            </label>
            <Select
              items={modes}
              value={mode}
              disabled={busy}
              onValueChange={(value) => {
                if (value === "APPEND" || value === "REPLACE") {
                  setMode(value);
                }
              }}
            >
              <SelectTrigger id="source-import-mode" className="w-52">
                <SelectValue placeholder="选择导入方式" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {modes.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <span className="text-xs text-muted-foreground">
              现有 {episodes.length} 集
            </span>
          </div>
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
        <DialogFooter className="shrink-0 gap-2">
          {preview && (
            <Button
              variant="ghost"
              disabled={busy}
              onClick={() => {
                setPreview(null);
                setDeletedPreview(null);
                setError("");
              }}
            >
              <ArrowLeft />
              返回原文
            </Button>
          )}
          <Button variant="outline" disabled={busy} onClick={onClose}>
            取消
          </Button>
          <Button
            onClick={preview ? submit : analyze}
            disabled={
              busy ||
              loadingExisting ||
              loadError ||
              !source.trim() ||
              preview?.segments.length === 0 ||
              !!preview?.segments.some((item) => !item.title.trim())
            }
          >
            {busy ? <Loader2 className="animate-spin" /> : <ListTree />}
            {busy
              ? "处理中"
              : preview
                ? `导入 ${preview.segments.length} 集`
                : "预览分集"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
