"use client";
import { useState } from "react";
import { Download, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { canvasTemplatesApi, type CanvasTemplate } from "@/lib/api/canvas-templates";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { useCanvas } from "../canvas-provider";

export function CanvasTemplateExportPanel({ onBusyChange }: { onBusyChange: (busy: boolean) => void }) {
  const snapshot = useCanvas(state => state.snapshot!);
  const saving = useCanvas(state => state.saving || !!state.pending);
  const [result, setResult] = useState<{ revision: number; template: CanvasTemplate } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloaded, setDownloaded] = useState(false);
  const fresh = result?.revision === snapshot.document.revision ? result.template : null;
  const inspect = async () => {
    setBusy(true); onBusyChange(true); setError(null); setDownloaded(false);
    try { setResult({ revision: snapshot.document.revision, template: await canvasTemplatesApi.export(snapshot.document.id, snapshot.document.revision) }); }
    catch (failure) { setError(getApiErrorMessage(failure)); }
    finally { setBusy(false); onBusyChange(false); }
  };
  const download = () => {
    if (!fresh) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(fresh)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = `${fresh.title.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").slice(0, 80)}.canvas.json`;
    document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000); setDownloaded(true);
  };
  return <section className="flex flex-col gap-5" aria-label="导出画布模板">
    <div className="flex items-start gap-3"><Layers className="size-6 shrink-0 text-primary" /><div className="flex flex-col gap-2">
      <h3 className="font-medium">把创作流程保存下来</h3>
      <p className="text-sm leading-relaxed text-muted-foreground">保存当前画布的节点、连线和固定结果{snapshot.document.projectId ? "，以及整个项目的剧本、分镜与资产" : "，以及引用的独立资产"}。导入后创建可独立编辑的新项目。</p>
    </div></div>
    <ul className="list-disc space-y-2 pl-5 text-sm text-muted-foreground">
      <li>模型和工具在导入时重新选择，服务凭据与原项目权限不会写入模板。</li>
      <li>媒体以待绑定清单保存。跨平台复用时，请同时从“导出文件”下载媒体文件。</li>
      <li>在画布列表点击“导入模板”，即可开始下一次创作。</li>
    </ul>
    {fresh && <dl className="grid grid-cols-2 gap-3 rounded-lg border border-border/30 bg-background/70 p-4 text-sm">
      <dt className="text-muted-foreground">节点</dt><dd className="text-right">{fresh.graph.nodes.length}</dd>
      <dt className="text-muted-foreground">项目内容</dt><dd className="text-right">{fresh.objects.length}</dd>
      <dt className="text-muted-foreground">固定结果</dt><dd className="text-right">{Object.keys(fresh.fixedOutputs).length}</dd>
      <dt className="text-muted-foreground">待选择的素材 / 服务</dt><dd className="text-right">{fresh.media.length} / {fresh.providers.length}</dd>
    </dl>}
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {downloaded && <p role="status" className="text-sm text-muted-foreground">模板文件已交给浏览器下载。</p>}
    <div className="flex justify-end gap-2">{fresh && <Button variant="outline" disabled={busy || saving} onClick={() => void inspect()}>重新检查</Button>}
      <Button disabled={busy || saving} onClick={() => fresh ? download() : void inspect()}>{fresh && <Download />}{busy ? "正在检查…" : fresh ? "下载模板" : "检查模板内容"}</Button>
    </div>
  </section>;
}
