"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Check,
  FileAudio,
  FileText,
  ImagePlus,
  Loader2,
  RefreshCw,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CanvasButton } from "../ui/canvas-button";
import { CanvasIconButton } from "../ui/canvas-icon-button";
import { CanvasInput } from "../ui/canvas-field";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import { MediaDeleteDialog } from "@/components/media/media-delete-dialog";
import { canvasUploadFile } from "@/lib/canvas/files";
import { MEDIA_LIBRARY_CHANGED } from "@/lib/media-library-events";
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
  surface = "default",
}: {
  kind?: ManagedMedia["content"]["kind"];
  value?: string;
  onSelect: (media: ManagedMedia) => void;
  disabled?: boolean;
  /** 画布悬浮面板与抽屉内使用画布紧凑圆角控件；弹窗沿用默认共享控件。 */
  surface?: "default" | "canvas";
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
  const [deleting, setDeleting] = useState<ManagedMedia | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const request = useRef(0);
  const [uploadRequest, setUploadRequest] = useState<{
    file: File;
    id: string;
  } | null>(null);
  const isCanvas = surface === "canvas";

  const load = useCallback(
    async (number: number, signal?: AbortSignal) => {
      const attempt = ++request.current;
      setLoading(true);
      setError(null);
      try {
        const result = await listManagedMedia(projectId, number, signal);
        if (attempt !== request.current || signal?.aborted) {
          return;
        }
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
        if (!signal?.aborted && attempt === request.current) {
          setError(getApiErrorMessage(failure));
        }
      } finally {
        if (!signal?.aborted && attempt === request.current) {
          setLoading(false);
        }
      }
    },
    [projectId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void load(1, controller.signal);
    return () => {
      controller.abort();
    };
  }, [load, mediaRevision]);

  useEffect(() => {
    const refresh = () => {
      void load(1);
    };
    window.addEventListener(MEDIA_LIBRARY_CHANGED, refresh);
    return () => {
      window.removeEventListener(MEDIA_LIBRARY_CHANGED, refresh);
    };
  }, [load]);

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
      if (kind && media.content.kind !== kind) {
        throw new Error(
          "文件实际类型与此节点不匹配，已保存在素材库，请选择相应节点使用",
        );
      }
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
    <section className="flex flex-col gap-2.5" aria-label="媒体素材库">
      <div className="flex items-center gap-2 p-0.5">
        {isCanvas ? (
          <div className="relative min-w-0 flex-1">
            <Search
              className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden
            />
            <CanvasInput
              name="canvas-media-search"
              aria-label="搜索已加载素材"
              placeholder="搜索素材"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              className="pl-8"
            />
          </div>
        ) : (
          <Input
            aria-label="搜索已加载素材"
            placeholder="搜索素材"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            className="min-w-0"
          />
        )}
        {isCanvas ? (
          <CanvasButton
            variant="secondary"
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
          </CanvasButton>
        ) : (
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
        )}
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
            if (file) {
              void upload(file);
            }
            event.target.value = "";
          }}
        />
      </div>
      {error && (
        <div
          role="alert"
          className="flex flex-col items-start gap-2 rounded-[10px] border border-destructive/20 bg-destructive/5 px-2.5 py-2 text-xs text-destructive"
        >
          <p>{error}</p>
          {uploadRequest &&
            (isCanvas ? (
              <CanvasButton
                variant="secondary"
                size="sm"
                disabled={uploading}
                onClick={() => void upload(uploadRequest.file)}
              >
                重试上传
              </CanvasButton>
            ) : (
              <Button
                variant="outline"
                size="sm"
                disabled={uploading}
                onClick={() => void upload(uploadRequest.file)}
              >
                重试上传
              </Button>
            ))}
        </div>
      )}
      <div className="grid grid-cols-2 gap-2.5 p-0.5">
        {visible.map((media) => {
          const active = value === media.id;
          return (
            <div key={media.id} className="group relative min-w-0">
              <button
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
                  "relative w-full overflow-hidden border text-left outline-none transition-[border-color,box-shadow,background-color] duration-150 focus-visible:ring-[3px] focus-visible:ring-primary/30 disabled:opacity-50 motion-reduce:transition-none",
                  isCanvas
                    ? "rounded-[12px] border-foreground/[0.08] bg-background/80 hover:border-foreground/[0.16] hover:shadow-sm dark:border-white/[0.1] dark:bg-white/[0.04] dark:hover:border-white/[0.18]"
                    : "rounded-xl border-border/30 bg-card/50",
                  active && "border-primary ring-2 ring-primary/20",
                )}
                aria-label={`选择${media.name}`}
                aria-pressed={active}
              >
                <div className="flex aspect-video items-center justify-center bg-foreground/[0.03] dark:bg-white/[0.03]">
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
                    <FileAudio className="size-7 text-muted-foreground" />
                  ) : (
                    <FileText className="size-7 text-muted-foreground" />
                  )}
                </div>
                <p
                  className={cn(
                    "truncate text-xs",
                    isCanvas ? "py-2 pl-2.5 pr-8" : "py-3 pl-3 pr-12",
                  )}
                >
                  {media.name}
                </p>
                {active && (
                  <span className="absolute right-1.5 top-1.5 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground shadow-xs">
                    <Check className="size-3" />
                  </span>
                )}
              </button>
              {canEdit &&
                (isCanvas ? (
                  <span className="absolute bottom-1.5 right-1.5 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(pointer:coarse)]:opacity-100 motion-reduce:transition-none">
                    <CanvasIconButton
                      variant="danger"
                      size="icon-xs"
                      label={`删除素材 ${media.name}`}
                      hint="删除素材"
                      disabled={disabled}
                      onClick={() => setDeleting(media)}
                    >
                      <Trash2 />
                    </CanvasIconButton>
                  </span>
                ) : (
                  <Button
                    variant="destructive-ghost"
                    size="icon-xs"
                    className="absolute bottom-2 right-2"
                    aria-label={`删除素材 ${media.name}`}
                    title="删除素材"
                    disabled={disabled}
                    onClick={() => setDeleting(media)}
                  >
                    <Trash2 />
                  </Button>
                ))}
            </div>
          );
        })}
      </div>
      {!visible.length && !loading && !error && (
        <div
          className={cn(
            "flex flex-col items-center gap-2 text-center text-xs text-muted-foreground",
            isCanvas
              ? "rounded-[12px] border border-dashed border-foreground/[0.1] bg-foreground/[0.02] px-4 py-6 dark:border-white/[0.1] dark:bg-white/[0.02]"
              : "items-start py-2 text-left text-sm",
          )}
        >
          {isCanvas && (
            <span className="grid size-8 place-items-center rounded-[10px] bg-foreground/[0.05] text-muted-foreground dark:bg-white/[0.07]">
              <ImagePlus className="size-4" aria-hidden />
            </span>
          )}
          <p>
            {search.trim()
              ? "没有找到匹配素材"
              : kind
                ? "还没有这类素材，可先上传一份"
                : "还没有素材，可上传图片、视频、音频或文档"}
          </p>
          {more && <p className="text-[11px]">还有更多素材，可点击下方继续加载</p>}
          {search.trim() &&
            (isCanvas ? (
              <CanvasButton
                variant="secondary"
                size="sm"
                onClick={() => setSearch("")}
              >
                清空搜索
              </CanvasButton>
            ) : (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSearch("")}
              >
                清空搜索
              </Button>
            ))}
        </div>
      )}
      {loading && (
        <p role="status" className="text-xs text-muted-foreground">
          正在读取素材…
        </p>
      )}
      <div className="flex items-center justify-between gap-2 pt-0.5">
        {more ? (
          isCanvas ? (
            <CanvasButton
              variant="secondary"
              size="sm"
              disabled={loading}
              onClick={() => void load(page + 1)}
            >
              加载更多
            </CanvasButton>
          ) : (
            <Button
              variant="outline"
              size="sm"
              disabled={loading}
              onClick={() => void load(page + 1)}
            >
              加载更多
            </Button>
          )
        ) : (
          <span />
        )}
        {isCanvas ? (
          <CanvasButton
            variant="ghost"
            size="sm"
            disabled={loading}
            onClick={() => void load(1)}
          >
            <RefreshCw />
            刷新素材
          </CanvasButton>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            disabled={loading}
            onClick={() => void load(1)}
          >
            刷新素材
          </Button>
        )}
      </div>
      {deleting && (
        <MediaDeleteDialog
          media={deleting}
          onClose={() => setDeleting(null)}
          onDeleted={() =>
            setItems((current) =>
              current.filter((item) => item.id !== deleting.id),
            )
          }
        />
      )}
    </section>
  );
}
