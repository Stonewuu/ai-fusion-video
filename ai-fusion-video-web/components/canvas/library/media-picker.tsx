"use client";
import { canvasUploadFile } from "@/lib/canvas/files";

import { useCallback, useEffect, useRef, useState } from "react";
import { Upload, FileAudio, FileText, Loader2, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import {
  listManagedMedia,
  uploadMedia,
  type ManagedMedia,
} from "@/lib/api/media";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCanvas } from "../canvas-provider";
import { useCanvasMediaImports } from "../imports/media-imports-provider";
import { cn } from "@/lib/utils";

export function CanvasMediaPicker({
  kind,
  value,
  onSelect,
  disabled = false,
}: {
  kind?: ManagedMedia["content"]["kind"];
  value?: string;
  onSelect: (media: ManagedMedia) => void;
  disabled?: boolean;
}) {
  const projectId = useCanvas((state) => state.snapshot!.document.projectId);
  const { mediaRevision } = useCanvasMediaImports();
  const canvasId = useCanvas((state) => state.canvasId);
  const canEdit = useCanvas(
    (state) =>
      state.snapshot!.permissions.canEdit &&
      !state.snapshot!.document.archived &&
      !state.pending,
  );
  const [items, setItems] = useState<ManagedMedia[]>([]);
  const [page, setPage] = useState(1);
  const [more, setMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const request = useRef(0);
  const [uploadRequest, setUploadRequest] = useState<{
    file: File;
    id: string;
  } | null>(null);
  const load = useCallback(
    async (number: number, signal?: AbortSignal) => {
      const attempt = ++request.current;
      setLoading(true);
      setError(null);
      try {
        const result = await listManagedMedia(projectId, number, signal);
        if (attempt !== request.current || signal?.aborted) return;
        setItems((current) =>
          number === 1
            ? result.items
            : [
                ...new Map(
                  [...current, ...result.items].map((item) => [item.id, item]),
                ).values(),
              ],
        );
        setPage(number);
        setMore(result.hasMore);
      } catch (failure) {
        if (!signal?.aborted && attempt === request.current)
          setError(getApiErrorMessage(failure));
      } finally {
        if (!signal?.aborted && attempt === request.current) setLoading(false);
      }
    },
    [projectId],
  );
  useEffect(() => {
    const controller = new AbortController();
    void load(1, controller.signal);
    return () => controller.abort();
  }, [load, mediaRevision]);
  const upload = async (file: File) => {
    setUploading(true);
    setError(null);
    const currentRequest =
      uploadRequest?.file === file
        ? uploadRequest
        : { file, id: crypto.randomUUID() };
    setUploadRequest(currentRequest);
    try {
      const media = await uploadMedia(canvasUploadFile(file), {
        requestId: currentRequest.id,
        canvasId,
        projectId: projectId ?? undefined,
      });
      setItems((current) => [
        media,
        ...current.filter((item) => item.id !== media.id),
      ]);
      setUploadRequest(null);
      if (kind && media.content.kind !== kind)
        throw new Error(
          "文件实际类型与此节点不匹配，已保存在素材库，请选择相应节点使用",
        );
      onSelect(media);
    } catch (failure) {
      setError(getApiErrorMessage(failure));
    } finally {
      setUploading(false);
    }
  };
  const visible = items.filter(
    (item) =>
      item.state === "READY" &&
      (!kind || item.content.kind === kind) &&
      item.name.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <section className="space-y-3" aria-label="媒体素材库">
      <div className="flex items-center gap-2">
        <Input
          aria-label="搜索已加载素材"
          placeholder="搜索素材"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          className="min-w-0"
        />
        <Button
          variant="outline"
          size="default"
          className="shrink-0"
          disabled={uploading || !canEdit || disabled}
          title="上传素材"
          aria-label="上传素材"
          onClick={() => input.current?.click()}
        >
          {uploading ? (
            <Loader2 className="animate-spin motion-reduce:animate-none" />
          ) : (
            <Upload />
          )}
          {uploading ? "上传中" : "上传"}
        </Button>
        <input
          ref={input}
          type="file"
          hidden
          accept={
            kind === "IMAGE"
              ? "image/*"
              : kind === "VIDEO"
                ? "video/*"
                : kind === "AUDIO"
                  ? "audio/*"
                  : kind === "TEXT"
                    ? "text/plain,text/markdown,text/csv,application/json,.md,.txt,.csv,.json"
                    : kind === "DOCUMENT"
                      ? "application/pdf"
                      : "image/*,video/*,audio/*,text/plain,text/markdown,text/csv,application/json,application/pdf,.md,.txt,.csv,.json"
          }
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
            event.target.value = "";
          }}
        />
      </div>
      {error && (
        <div role="alert" className="space-y-2 text-sm text-destructive">
          <p>{error}</p>
          {uploadRequest && (
            <Button
              variant="outline"
              size="sm"
              disabled={uploading}
              onClick={() => void upload(uploadRequest.file)}
            >
              重试上传
            </Button>
          )}
        </div>
      )}
      <div className="grid grid-cols-2 gap-3">
        {visible.map((media) => (
          <button
            key={media.id}
            type="button"
            onClick={() => onSelect(media)}
            disabled={disabled}
            draggable={canEdit && !disabled}
            onDragStart={(event) => {
              event.dataTransfer.setData(
                "application/x-fusion-media",
                JSON.stringify(media),
              );
              event.dataTransfer.effectAllowed = "copy";
            }}
            className={cn(
              "relative overflow-hidden rounded-xl border border-border/30 bg-card/50 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
              value === media.id && "border-primary bg-primary/5",
            )}
            aria-label={`选择${media.name}`}
            aria-pressed={value === media.id}
          >
            <div className="flex aspect-video items-center justify-center bg-muted/30">
              {media.content.kind === "IMAGE" ||
              media.content.kind === "VIDEO" ? (
                <MediaThumbnail
                  source={`media://${media.id}`}
                  kind={media.content.kind === "IMAGE" ? "image" : "video"}
                  alt={media.name}
                  fit="cover"
                  interactive={false}
                />
              ) : media.content.kind === "AUDIO" ? (
                <FileAudio className="size-8 text-muted-foreground" />
              ) : (
                <FileText className="size-8 text-muted-foreground" />
              )}
            </div>
            <p className="truncate px-3 py-2 text-xs">{media.name}</p>
            {value === media.id && (
              <Check className="absolute right-2 top-2 size-4 text-primary" />
            )}
          </button>
        ))}
      </div>
      {!visible.length && !loading && !error && (
        <div className="flex flex-col items-start gap-2 text-sm text-muted-foreground">
          <p>{search.trim() ? "没有找到匹配素材" : kind ? "还没有这类素材，可先上传一份。" : "还没有素材，可上传图片、视频、音频或文档。"}</p>
          {more && <p className="text-xs">还有更多素材，可点击下方“加载更多”继续查看。</p>}
          {search.trim() && <Button variant="outline" size="sm" onClick={() => setSearch("")}>清空搜索</Button>}
        </div>
      )}
      {loading && (
        <p role="status" className="text-sm text-muted-foreground">
          正在读取素材…
        </p>
      )}
      <div className="flex gap-2">
        {more && (
          <Button
            variant="outline"
            size="sm"
            disabled={loading}
            onClick={() => void load(page + 1)}
          >
            加载更多
          </Button>
        )}
        <Button
          variant="ghost"
          size="sm"
          disabled={loading}
          onClick={() => void load(1)}
        >
          刷新素材
        </Button>
      </div>
    </section>
  );
}
