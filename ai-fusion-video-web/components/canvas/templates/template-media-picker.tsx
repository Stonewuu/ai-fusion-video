"use client";
import { useEffect, useRef, useState } from "react";
import { Check, FileText, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MediaThumbnail } from "@/components/media/media-thumbnail";
import { listManagedMedia, uploadMedia, type ManagedMedia } from "@/lib/api/media";
import { projectApi, type Project } from "@/lib/api/project";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { canvasUploadFile } from "@/lib/canvas/files";
import { CanvasSelect } from "../forms/canvas-select";
import type { TemplateMedia } from "@/lib/api/canvas-templates";
import { cn } from "@/lib/utils";

export function TemplateMediaPicker({ slot, value, onSelect, disabled, onBusyChange }: {
  slot: TemplateMedia; value?: string; onSelect: (media: ManagedMedia) => void; disabled: boolean; onBusyChange: (busy: boolean) => void;
}) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [scope, setScope] = useState("personal");
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [data, setData] = useState<{ key: string; items: ManagedMedia[]; more: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [pending, setPending] = useState<{ file: File; id: string } | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const key = `${scope}:${page}`;
  useEffect(() => {
    let active = true;
    void projectApi.list().then(items => { if (active) setProjects(items); }).catch(failure => { if (active) setError(getApiErrorMessage(failure)); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    void listManagedMedia(scope === "personal" ? null : Number(scope), page, controller.signal)
      .then(result => { if (!controller.signal.aborted) { setData({ key, items: result.items, more: result.hasMore }); setError(null); } })
      .catch(failure => { if (!controller.signal.aborted) setError(getApiErrorMessage(failure)); });
    return () => controller.abort();
  }, [scope, page, key, attempt]);
  const upload = async (file: File) => {
    const request = pending?.file === file ? pending : { file, id: crypto.randomUUID() };
    setPending(request); setUploading(true); onBusyChange(true); setError(null);
    try {
      const media = await uploadMedia(canvasUploadFile(file), { requestId: request.id });
      setPending(null);
      if (media.content.kind !== slot.content.kind) throw new Error("文件实际类型不匹配；已上传的文件保留在素材库，请选择正确类型。");
      onSelect(media);
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setUploading(false); onBusyChange(false); }
  };
  const visible = data?.key === key ? data.items.filter(item => item.content.kind === slot.content.kind && item.name.toLowerCase().includes(search.toLowerCase())) : [];
  return <div className="flex flex-col gap-3 rounded-lg border border-border/30 bg-background/70 p-3">
    <CanvasSelect label="素材所在位置" value={scope} placeholder="选择素材位置" disabled={disabled || uploading}
      items={[{ value: "personal", label: "当前空间素材库" }, ...projects.map(item => ({ value: String(item.id), label: item.name }))]}
      onChange={value => { setScope(value); setPage(1); }} />
    <div className="flex gap-2"><Input aria-label="搜索本页素材" placeholder="搜索本页素材" value={search} onChange={event => setSearch(event.target.value)} />
      <Button variant="outline" disabled={disabled || uploading} onClick={() => input.current?.click()}><Upload />{uploading ? "上传中" : "上传文件"}</Button></div>
    <input ref={input} type="file" hidden accept={slot.content.kind === "IMAGE" ? "image/*" : slot.content.kind === "VIDEO" ? "video/*" : slot.content.kind === "AUDIO" ? "audio/*" : slot.content.kind === "DOCUMENT" ? ".pdf" : ".txt,.md,.csv,.json"}
      onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void upload(file); }} />
    {error && <div className="flex flex-col gap-2"><p role="alert" className="text-sm text-destructive">{error}</p>
      {!pending && <Button variant="outline" size="sm" disabled={uploading} onClick={() => { setError(null); setAttempt(value => value + 1); }}>重新读取素材</Button>}</div>}
    {pending && !uploading && <Button variant="outline" onClick={() => void upload(pending.file)}>重试原文件上传</Button>}
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {visible.map(media => <button key={media.id} type="button" disabled={disabled || uploading} aria-pressed={value === media.id} aria-label={`选择 ${media.name}`} onClick={() => onSelect(media)}
        className={cn("flex min-w-0 flex-col gap-2 rounded-xl border border-border/30 bg-card p-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50", value === media.id && "border-primary bg-primary/5")}>
        <span className="flex aspect-video w-full items-center justify-center overflow-hidden rounded-lg bg-muted/20">
          {media.content.kind === "IMAGE" || media.content.kind === "VIDEO" ? <MediaThumbnail source={`media://${media.id}`} kind={media.content.kind === "IMAGE" ? "image" : "video"} alt={media.name} interactive={false} /> : <FileText className="size-6 text-muted-foreground" />}
        </span><span className="flex w-full items-center gap-1 text-xs"><span className="truncate">{media.name}</span>{value === media.id && <Check className="size-4 shrink-0 text-primary" />}</span>
      </button>)}
    </div>
    {data?.key !== key ? !error && <p role="status" className="text-sm text-muted-foreground">读取素材中…</p> : !visible.length && <p className="text-sm text-muted-foreground">本页没有匹配素材，可以上传文件或切换位置。</p>}
    <div className="flex items-center justify-between gap-2"><span className="text-xs text-muted-foreground">上传的文件保存在当前空间素材库。</span>
      <div className="flex gap-2"><Button variant="ghost" size="xs" disabled={page === 1 || uploading} onClick={() => setPage(page - 1)}>上一页</Button>
        <Button variant="ghost" size="xs" disabled={data?.key !== key || !data.more || uploading} onClick={() => setPage(page + 1)}>下一页</Button></div></div>
  </div>;
}
