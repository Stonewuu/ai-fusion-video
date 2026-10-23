"use client";
import { useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { FileUp, Check, ImageIcon, FileText, FileVideo, FileAudio, Layers, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CanvasDialogContent as DialogContent } from "@/components/canvas/shared/canvas-dialog";
import { canvasTemplatesApi, type CanvasTemplate, type TemplateBinding, type TemplateBindings, type TemplateImportRequest, type TemplatePreview, type TemplateProvider } from "@/lib/api/canvas-templates";
import { canvasApi } from "@/lib/api/canvas";
import { getApiErrorMessage, getApiErrorStatus } from "@/lib/api/api-error";
import { useAuthStore } from "@/lib/store/auth-store";
import { clearTemplatePending, readTemplatePending, saveTemplatePending } from "@/lib/canvas/template-pending";
import { TemplateProviderFields } from "./template-provider-fields";
import { TemplateMediaPicker } from "./template-media-picker";

const MEDIA_ICONS = { IMAGE: ImageIcon, VIDEO: FileVideo, AUDIO: FileAudio, TEXT: FileText, DOCUMENT: FileText };

export function CanvasTemplateImportDialog({ onClose }: { onClose: () => void }) {
  const userId = useAuthStore(state => state.user!.id);
  return <TemplateImportDialog key={userId} userId={userId} onClose={onClose} />;
}
function TemplateImportDialog({ userId, onClose }: { userId: number; onClose: () => void }) {
  const router = useRouter();
  const key = `canvas-template:${userId}`;
  const field = useId();
  const fileInput = useRef<HTMLInputElement>(null);
  const [loaded, setLoaded] = useState(false);
  const [storageAttempt, setStorageAttempt] = useState(0);
  const [pending, setPending] = useState<TemplateImportRequest | null>(null);
  const [template, setTemplate] = useState<CanvasTemplate | null>(null);
  const [templateJson, setTemplateJson] = useState("");
  const [filename, setFilename] = useState("");
  const [title, setTitle] = useState("");
  const [bindings, setBindings] = useState<TemplateBindings>({ media: {}, providers: {} });
  const [names, setNames] = useState<Record<string, string>>({});
  const [selecting, setSelecting] = useState<string | null>(null);
  const [preview, setPreview] = useState<TemplatePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const blocked = busy || uploading || !loaded;
  useEffect(() => {
    let active = true;
    void readTemplatePending(key).then(request => { if (active) { setPending(request); setLoaded(true); setError(null); } })
      .catch(failure => { if (active) setError(getApiErrorMessage(failure)); });
    return () => { active = false; };
  }, [key, storageAttempt]);
  const chooseFile = async (file: File) => {
    setBusy(true); setError(null);
    try {
      if (file.size > 16 * 1024 * 1024) throw new Error("模板文件不能超过 16 MiB，请拆分画布后重新导出。");
      const content = await file.text();
      const inspected = await canvasTemplatesApi.inspect(content);
      setTemplate(inspected); setTemplateJson(JSON.stringify(inspected)); setFilename(file.name); setTitle(inspected.title);
      setBindings({ media: {}, providers: {} }); setNames({}); setPreview(null); setSelecting(null);
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  const providerChange = (slot: TemplateProvider, binding: TemplateBinding) => {
    const providers = { ...bindings.providers, [slot.id]: binding };
    if (slot.kind === "COMFYUI_MODEL") for (const item of template!.providers) {
      if (item.kind === "WORKFLOW" && item.uses.some(use => slot.uses.some(other => use.nodeId === other.nodeId && use.batch === other.batch))) delete providers[item.id];
    }
    setBindings({ ...bindings, providers }); setPreview(null);
  };
  const complete = !!template && template.media.every(item => bindings.media[item.id]) && template.providers.every(item => bindings.providers[item.id]);
  const inspect = async () => {
    setBusy(true); setError(null);
    try { setPreview(await canvasTemplatesApi.preview(templateJson, bindings)); }
    catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  const openCreated = async (id: number, canvasId: string) => {
    await clearTemplatePending(key); setPending(null); router.push(`/projects/${id}/canvases/${canvasId}`); onClose();
  };
  const submit = async () => {
    if (!pending && !preview) return;
    setBusy(true); setError(null);
    try {
      await navigator.locks.request(key, { ifAvailable: true }, async lock => {
        if (!lock) throw new Error("另一窗口正在导入模板，请等待原请求完成。");
        const stored = await readTemplatePending(key);
        if (stored && stored.requestId !== pending?.requestId) {
          setPending(stored); throw new Error("已有待确认的模板导入，请先确认启动状态。");
        }
        const request = pending ?? { requestId: crypto.randomUUID(), canvasId: crypto.randomUUID(), title: title.trim(), templateJson, bindings, previewHash: preview!.hash };
        await saveTemplatePending(key, request); setPending(request);
        try {
          const result = await canvasTemplatesApi.import(request);
          await openCreated(result.result.id, result.result.canvasId);
        } catch (failure) {
          if ([400, 409, 422].includes(getApiErrorStatus(failure) ?? 0)) {
            try {
              const found = await canvasApi.snapshot(request.canvasId);
              if (found.document.projectId) { await openCreated(found.document.projectId, found.document.id); return; }
            } catch (lookupFailure) {
              if (getApiErrorStatus(lookupFailure) === 404) { await clearTemplatePending(key); setPending(null); setPreview(null); }
            }
          }
          throw failure;
        }
      });
    } catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); }
  };
  return <Dialog open onOpenChange={open => { if (!open && !busy && !uploading) onClose(); }}>
    <DialogContent className="flex max-h-[calc(100vh-2rem)] flex-col overflow-hidden sm:max-w-2xl">
      <DialogHeader className="shrink-0"><DialogTitle className="flex items-center gap-2"><Layers className="size-5 text-primary" />从模板创建项目</DialogTitle>
        <DialogDescription>导入画布、剧本与分镜，选择素材和服务后创建独立项目。</DialogDescription></DialogHeader>
      <div className="flex min-h-0 flex-col gap-6 overflow-y-auto px-1 py-2">
        {pending ? <div className="flex flex-col gap-3 rounded-xl border border-primary/30 bg-primary/5 p-4" role="status">
          <p className="font-medium">{pending.title}</p><p className="text-sm text-muted-foreground">原导入请求尚待确认。继续会查询或完成同一次操作，关闭或刷新后也可回来继续。</p>
        </div> : <>
          <section className="flex flex-col gap-3">
            <div className="flex items-center justify-between gap-3"><span className="text-sm font-medium">1 · 选择模板</span>
              <Button variant="outline" disabled={blocked} onClick={() => fileInput.current?.click()}><FileUp />{template ? "更换文件" : "选择文件"}</Button></div>
            <input ref={fileInput} type="file" accept=".json,application/json" hidden onChange={event => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void chooseFile(file); }} />
            {template ? <div className="flex flex-col gap-2 rounded-lg border border-border/30 bg-background/70 p-3"><p className="break-all text-sm">{filename}</p>
              <p className="text-xs text-muted-foreground">{template.graph.nodes.length} 个节点 · {template.objects.length} 个项目内容 · {Object.keys(template.fixedOutputs).length} 份固定结果</p></div>
              : <p className="text-sm text-muted-foreground">使用“导出与交付 → 画布模板”保存的 .canvas.json 文件，最大 16 MiB。</p>}
          </section>
          {template && <>
            <section className="flex flex-col gap-4"><h3 className="text-sm font-medium">2 · 准备创作内容</h3>
              {template.providers.length > 0 && <TemplateProviderFields template={template} bindings={bindings} disabled={blocked} onChange={providerChange} />}
              {template.media.length > 0 && <div className="flex flex-col gap-3"><p className="text-sm text-muted-foreground">模板不内嵌媒体文件。请选择已有素材，或上传交付包中的文件。</p>
                {template.media.map(slot => { const Icon = MEDIA_ICONS[slot.content.kind]; return <div key={slot.id} className="flex flex-col gap-3 border-b border-border/20 pb-3 last:border-0 last:pb-0">
                  <div className="flex items-center gap-3"><Icon className="size-4 shrink-0 text-primary" /><div className="min-w-0 flex-1">
                    <p className="truncate text-sm" title={slot.name}>{slot.name}</p><p className="truncate text-xs text-muted-foreground" title={names[slot.id]}>{names[slot.id] ? `已选：${names[slot.id]}` : "尚未选择素材"}</p></div>
                    {bindings.media[slot.id] && <Check className="size-4 text-primary" />}<Button variant="outline" size="sm" disabled={blocked} onClick={() => setSelecting(selecting === slot.id ? null : slot.id)}>{selecting === slot.id ? "收起" : bindings.media[slot.id] ? "更换" : "选择"}</Button></div>
                  {selecting === slot.id && <TemplateMediaPicker slot={slot} value={bindings.media[slot.id]?.id} disabled={busy} onBusyChange={setUploading} onSelect={media => {
                    setBindings({ ...bindings, media: { ...bindings.media, [slot.id]: { id: media.id, version: media.version } } });
                    setNames({ ...names, [slot.id]: media.name }); setPreview(null); setSelecting(null);
                  }} />}
                </div>; })}
              </div>}
              {!template.media.length && !template.providers.length && <p className="text-sm text-muted-foreground">内容已齐全，无需选择额外素材或服务。</p>}
            </section>
            <section className="flex flex-col gap-4"><h3 className="text-sm font-medium">3 · 确认新项目</h3>
              <div className="flex flex-col gap-2"><Label htmlFor={field}>项目名称</Label><Input id={field} value={title} maxLength={200} disabled={blocked} onChange={event => setTitle(event.target.value)} /></div>
              {preview && <div role="status" className="flex flex-col gap-3 rounded-xl border border-primary/30 bg-primary/5 p-4">
                <p className="text-sm font-medium">保存到 {preview.ownership.name} · {preview.ownership.type === 2 ? "团队" : "个人"}</p>
                <p className="text-sm text-muted-foreground">{preview.nodes} 个节点、{preview.businessObjects} 个项目内容和 {preview.mediaFiles} 个素材已检查。</p>
                {preview.ownership.type === 2 && <p className="text-sm text-muted-foreground">团队成员可读取新项目及其导入的素材。</p>}
                {preview.notices.map(notice => <p key={notice} className="text-sm text-amber-700 dark:text-amber-300">{notice}</p>)}
              </div>}
              <p className="text-xs text-muted-foreground">导入后可独立编辑。生成操作由你在新画布中启动。</p>
            </section>
          </>}
        </>}
        {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        {!loaded && error && <Button variant="outline" onClick={() => { setError(null); setStorageAttempt(value => value + 1); }}>重试读取本地记录</Button>}
      </div>
      <DialogFooter className="shrink-0"><Button variant="outline" disabled={busy || uploading} onClick={onClose}>{pending ? "稍后继续" : "取消"}</Button>
        <Button disabled={blocked || (!pending && (!complete || !title.trim()))} onClick={() => void (pending || preview ? submit() : inspect())}>
          {busy && <Loader2 className="animate-spin motion-reduce:animate-none" />}{pending ? "确认启动状态" : preview ? "创建项目" : "检查导入内容"}
        </Button></DialogFooter>
    </DialogContent>
  </Dialog>;
}
