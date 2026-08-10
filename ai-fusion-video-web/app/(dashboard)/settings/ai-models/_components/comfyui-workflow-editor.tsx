"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Copy, FileJson2, Loader2, Save, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useConfirm } from "@/components/ui/confirm-dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { ApiConfig } from "@/lib/api/ai-model";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  comfyUiWorkflowApi,
  type ComfyUiInputBindings,
  type ComfyUiOutputBinding,
  type ComfyUiWorkflow,
  type ComfyUiWorkflowModelType,
  type ComfyUiWorkflowVersion,
  type ComfyUiWorkflowVersionSaveReq,
} from "@/lib/api/comfyui-workflow";
import { cn } from "@/lib/utils";
import { ComfyUiBindingEditor } from "./comfyui-binding-editor";
import { ComfyUiVersionActions } from "./comfyui-version-actions";
import {
  formatComfyUiNodeLabel,
  parseComfyUiNodes,
  parseInputBindingsJson,
  parseOutputBindingsJson,
} from "./comfyui-workflow-support";

interface ComfyUiWorkflowEditorProps {
  workflow: ComfyUiWorkflow | null;
  apiConfig: ApiConfig;
  onWorkflowCreated: (id: number) => Promise<void> | void;
  onWorkflowChanged: () => Promise<void> | void;
}

interface VersionDraft {
  id?: number;
  uiWorkflowJson: string;
  apiWorkflowJson: string;
  inputBindings: ComfyUiInputBindings;
  outputBindings: ComfyUiOutputBinding[];
}

const STEPS = ["基本信息", "导入工作流", "检查节点", "输入绑定", "输出绑定", "验证发布"] as const;
const EMPTY_VERSION: VersionDraft = {
  uiWorkflowJson: "",
  apiWorkflowJson: "",
  inputBindings: {},
  outputBindings: [],
};

function versionToDraft(version: ComfyUiWorkflowVersion): VersionDraft {
  return {
    id: version.id,
    uiWorkflowJson: version.uiWorkflowJson || "",
    apiWorkflowJson: version.apiWorkflowJson,
    inputBindings: parseInputBindingsJson(version.inputBindingsJson),
    outputBindings: parseOutputBindingsJson(version.outputBindingsJson),
  };
}

export function ComfyUiWorkflowEditor({
  workflow,
  apiConfig,
  onWorkflowCreated,
  onWorkflowChanged,
}: ComfyUiWorkflowEditorProps) {
  const { confirm } = useConfirm();
  const apiFileRef = useRef<HTMLInputElement>(null);
  const uiFileRef = useRef<HTMLInputElement>(null);
  const [step, setStep] = useState(0);
  const [saving, setSaving] = useState(false);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [versions, setVersions] = useState<ComfyUiWorkflowVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<ComfyUiWorkflowVersion | null>(null);
  const [versionDraft, setVersionDraft] = useState<VersionDraft>(EMPTY_VERSION);
  const [metadata, setMetadata] = useState({
    apiConfigId: apiConfig.id,
    name: "",
    code: "",
    modelType: 2 as ComfyUiWorkflowModelType,
    description: "",
    status: 1,
  });

  const loadVersions = useCallback(async (preferredVersionId?: number) => {
    if (!workflow) {
      setVersions([]);
      setSelectedVersion(null);
      setVersionDraft(EMPTY_VERSION);
      return;
    }
    setVersionsLoading(true);
    try {
      const items = await comfyUiWorkflowApi.versions(workflow.id);
      setVersions(items);
      const preferredId = preferredVersionId ?? selectedVersion?.id ?? workflow.activeVersionId;
      const next = items.find(item => item.id === preferredId) || items[0] || null;
      setSelectedVersion(next);
      setVersionDraft(next ? versionToDraft(next) : EMPTY_VERSION);
    } catch (error) {
      toastApiError(error, "加载工作流版本失败");
    } finally {
      setVersionsLoading(false);
    }
  }, [selectedVersion?.id, workflow]);

  useEffect(() => {
    setStep(0);
    setMetadata({
      apiConfigId: apiConfig.id,
      name: workflow?.name || "",
      code: workflow?.code || "",
      modelType: workflow?.modelType || 2,
      description: workflow?.description || "",
      status: workflow?.status ?? 1,
    });
    void loadVersions(workflow?.activeVersionId ?? undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiConfig.id, workflow?.id]);

  const parsedNodes = useMemo(() => {
    try {
      return { nodes: parseComfyUiNodes(versionDraft.apiWorkflowJson), error: "" };
    } catch (error) {
      return { nodes: [], error: error instanceof Error ? error.message : "工作流解析失败" };
    }
  }, [versionDraft.apiWorkflowJson]);

  const saveMetadata = async () => {
    if (!metadata.apiConfigId || !metadata.name.trim() || !metadata.code.trim()) return;
    setSaving(true);
    try {
      if (workflow) {
        await comfyUiWorkflowApi.update({ id: workflow.id, ...metadata });
        toast.success("工作流信息已保存");
        await onWorkflowChanged();
      } else {
        const id = await comfyUiWorkflowApi.create(metadata);
        toast.success("工作流已创建", { description: "下一步导入 API-format JSON。" });
        await onWorkflowCreated(id);
      }
    } catch (error) {
      toastApiError(error, "保存工作流失败");
    } finally {
      setSaving(false);
    }
  };

  const saveVersion = async () => {
    if (!workflow || !versionDraft.apiWorkflowJson.trim()) return;
    setSaving(true);
    const payload: ComfyUiWorkflowVersionSaveReq = {
      id: versionDraft.id,
      workflowId: workflow.id,
      uiWorkflowJson: versionDraft.uiWorkflowJson,
      apiWorkflowJson: versionDraft.apiWorkflowJson,
      inputBindingsJson: JSON.stringify(versionDraft.inputBindings),
      outputBindingsJson: JSON.stringify(versionDraft.outputBindings),
    };
    try {
      let versionId = versionDraft.id;
      if (versionId) {
        await comfyUiWorkflowApi.updateVersion(payload);
      } else {
        versionId = await comfyUiWorkflowApi.createVersion(payload);
      }
      toast.success("工作流版本已保存");
      await loadVersions(versionId);
    } catch (error) {
      toastApiError(error, "保存工作流版本失败");
    } finally {
      setSaving(false);
    }
  };

  const deleteDraftVersion = async () => {
    if (!selectedVersion || selectedVersion.published) return;
    const accepted = await confirm({
      title: "删除工作流草稿",
      description: `确定删除 v${selectedVersion.versionNo} 草稿吗？此操作不可撤销。`,
      confirmText: "删除草稿",
      variant: "destructive",
    });
    if (!accepted) return;
    try {
      await comfyUiWorkflowApi.deleteVersion(selectedVersion.id);
      await loadVersions();
      toast.success("工作流草稿已删除");
    } catch (error) {
      toastApiError(error, "删除工作流草稿失败");
    }
  };

  const importJson = async (file: File, kind: "api" | "ui") => {
    const value = await file.text();
    try {
      JSON.parse(value);
      setVersionDraft(previous => kind === "api"
        ? { ...previous, apiWorkflowJson: value }
        : { ...previous, uiWorkflowJson: value });
      toast.success(kind === "api" ? "API-format JSON 已导入" : "UI-format JSON 已导入");
    } catch {
      toast.error("文件不是合法 JSON");
    }
  };

  const versionLocked = Boolean(selectedVersion?.published && versionDraft.id === selectedVersion.id);
  const versionOptions = versions.map(version => ({
    value: String(version.id),
    label: `v${version.versionNo}${workflow?.activeVersionId === version.id ? " · 当前发布" : version.published ? " · 历史发布" : " · 草稿"}`,
  }));

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 border-b border-border/20 pb-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-sm font-semibold">{workflow?.name || "新建 ComfyUI 工作流"}</p>
            <p className="mt-1 text-xs text-muted-foreground">按 6 步完成导入、绑定、试运行与发布。</p>
          </div>
          {workflow && (
            <div className="flex flex-col items-end gap-1.5">
              <div className="flex items-center gap-2">
                <Select
                  value={selectedVersion ? String(selectedVersion.id) : null}
                  onValueChange={value => {
                    const version = versions.find(item => item.id === Number(value)) || null;
                    setSelectedVersion(version);
                    setVersionDraft(version ? versionToDraft(version) : EMPTY_VERSION);
                  }}
                  items={versionOptions}
                  disabled={versionsLoading || versions.length === 0}
                >
                  <SelectTrigger className="w-44 text-xs"><SelectValue placeholder={versionsLoading ? "加载中" : versions.length === 0 ? "暂无版本" : "选择版本"} /></SelectTrigger>
                  <SelectContent className="text-xs"><SelectGroup>{versionOptions.map(option => <SelectItem key={option.value} value={option.value} className="text-xs">{option.label}</SelectItem>)}</SelectGroup></SelectContent>
                </Select>
                <Button variant="outline" onClick={() => { setSelectedVersion(null); setVersionDraft({ ...versionDraft, id: undefined }); setStep(1); }}>
                  <Copy />{selectedVersion ? "复制为新版本" : "新建版本"}
                </Button>
                {selectedVersion && !selectedVersion.published && (
                  <Button variant="destructive-ghost" size="icon" aria-label="删除工作流草稿" onClick={() => void deleteDraftVersion()}>
                    <Trash2 />
                  </Button>
                )}
              </div>
              {!versionsLoading && versions.length === 0 && (
                <p className="text-xs text-muted-foreground">暂无工作流版本，请点击“新建版本”开始导入。</p>
              )}
            </div>
          )}
        </div>

        <div className="mt-3 flex gap-1 overflow-x-auto pb-1">
          {STEPS.map((label, index) => (
            <button
              key={label}
              type="button"
              onClick={() => workflow || index === 0 ? setStep(index) : undefined}
              disabled={!workflow && index > 0}
              className={cn(
                "h-8 shrink-0 rounded-xl px-3 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-40",
                step === index ? "bg-primary text-primary-foreground" : "bg-muted/40 text-muted-foreground hover:text-foreground",
              )}
            >
              {index + 1}. {label}
            </button>
          ))}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-1 py-4">
        {step === 0 && (
          <div className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-1.5"><Label className="text-xs text-muted-foreground">名称</Label><Input value={metadata.name} onChange={event => setMetadata(previous => ({ ...previous, name: event.target.value }))} placeholder="例如：SDXL 文生图" /></div>
              <div className="space-y-1.5"><Label className="text-xs text-muted-foreground">稳定标识</Label><Input value={metadata.code} onChange={event => setMetadata(previous => ({ ...previous, code: event.target.value }))} placeholder="sdxl_text_to_image" className="font-mono" /></div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">所属 ComfyUI 供应商</Label>
                <Input value={apiConfig.name} readOnly aria-label="所属 ComfyUI 供应商" />
              </div>
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">生成类型</Label>
                <Select value={metadata.modelType} onValueChange={value => setMetadata(previous => ({ ...previous, modelType: Number(value) as ComfyUiWorkflowModelType }))} items={[{ value: 2, label: "图片生成" }, { value: 3, label: "视频生成" }]}>
                  <SelectTrigger className="w-full"><SelectValue placeholder="选择类型" /></SelectTrigger>
                  <SelectContent><SelectGroup><SelectItem value={2}>图片生成</SelectItem><SelectItem value={3}>视频生成</SelectItem></SelectGroup></SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-1.5"><Label className="text-xs text-muted-foreground">说明</Label><Textarea value={metadata.description} onChange={event => setMetadata(previous => ({ ...previous, description: event.target.value }))} placeholder="说明适用模型、必要自定义节点与输入要求" className="min-h-24 resize-none" /></div>
            <div className="flex justify-between gap-2">
              <Button variant="secondary" size="sm" onClick={() => setMetadata(previous => ({ ...previous, status: previous.status === 1 ? 0 : 1 }))}>
                {metadata.status === 1 ? "设为禁用" : "设为启用"}
              </Button>
              <Button size="sm" onClick={saveMetadata} disabled={saving || !metadata.apiConfigId || !metadata.name.trim() || !metadata.code.trim()}>{saving ? <Loader2 className="animate-spin" /> : <Save />}{workflow ? "保存信息" : "创建并继续"}</Button>
            </div>
          </div>
        )}

        {step === 1 && workflow && (
          <div className="space-y-4">
            {versionLocked && <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-700 dark:text-amber-300">已发布版本不可修改。点击“复制为新版本”后再编辑。</div>}
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3"><div><Label>API-format JSON</Label><p className="mt-1 text-xs text-muted-foreground">在 ComfyUI 中选择 Save (API Format) 或 Export (API)。</p></div><Button variant="outline" size="sm" onClick={() => apiFileRef.current?.click()} disabled={versionLocked}><FileJson2 />导入 API JSON</Button></div>
              <input ref={apiFileRef} type="file" accept="application/json,.json" className="hidden" onChange={event => { const file = event.target.files?.[0]; if (file) void importJson(file, "api"); event.target.value = ""; }} />
              <Textarea value={versionDraft.apiWorkflowJson} onChange={event => setVersionDraft(previous => ({ ...previous, apiWorkflowJson: event.target.value }))} readOnly={versionLocked} spellCheck={false} className="min-h-72 resize-y font-mono text-xs" placeholder='{"1":{"class_type":"KSampler","inputs":{...}}}' />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3"><div><Label>UI-format JSON（可选）</Label><p className="mt-1 text-xs text-muted-foreground">仅用于留档和回到 ComfyUI 编辑，不参与执行。</p></div><Button variant="outline" size="sm" onClick={() => uiFileRef.current?.click()} disabled={versionLocked}><FileJson2 />导入 UI JSON</Button></div>
              <input ref={uiFileRef} type="file" accept="application/json,.json" className="hidden" onChange={event => { const file = event.target.files?.[0]; if (file) void importJson(file, "ui"); event.target.value = ""; }} />
              <Textarea value={versionDraft.uiWorkflowJson} onChange={event => setVersionDraft(previous => ({ ...previous, uiWorkflowJson: event.target.value }))} readOnly={versionLocked} spellCheck={false} className="min-h-32 resize-y font-mono text-xs" placeholder="可留空" />
            </div>
          </div>
        )}

        {step === 2 && workflow && (
          <div className="space-y-3">
            {parsedNodes.error ? <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">{parsedNodes.error}</div> : <p className="text-sm text-muted-foreground">解析到 {parsedNodes.nodes.length} 个节点。绑定前请确认节点类型与目标 ComfyUI 实例一致。</p>}
            <div className="grid gap-2 md:grid-cols-2">{parsedNodes.nodes.map(node => <div key={node.id} className="rounded-lg border border-border/20 bg-background/70 p-3"><p className="font-mono text-xs font-medium">{formatComfyUiNodeLabel(node)}</p><p className="mt-1 text-[11px] text-muted-foreground">{node.inputNames.join("、") || "无输入"}</p></div>)}</div>
          </div>
        )}

        {(step === 3 || step === 4) && workflow && (
          <ComfyUiBindingEditor mode={step === 3 ? "input" : "output"} modelType={metadata.modelType} apiWorkflowJson={versionDraft.apiWorkflowJson} inputBindings={versionDraft.inputBindings} outputBindings={versionDraft.outputBindings} onInputBindingsChange={inputBindings => setVersionDraft(previous => ({ ...previous, inputBindings }))} onOutputBindingsChange={outputBindings => setVersionDraft(previous => ({ ...previous, outputBindings }))} />
        )}

        {step === 5 && workflow && <ComfyUiVersionActions workflow={workflow} version={selectedVersion} onUpdated={async () => { await loadVersions(selectedVersion?.id); await onWorkflowChanged(); }} />}
      </div>

      {workflow && step > 0 && step < 5 && (
        <div className="flex shrink-0 justify-between gap-2 border-t border-border/20 pt-3">
          <Button variant="outline" size="sm" onClick={() => setStep(value => Math.max(0, value - 1))}>上一步</Button>
          <div className="flex gap-2"><Button variant="secondary" size="sm" onClick={saveVersion} disabled={saving || versionLocked || !versionDraft.apiWorkflowJson.trim()}>{saving ? <Loader2 className="animate-spin" /> : <Save />}保存版本</Button><Button size="sm" onClick={() => setStep(value => Math.min(5, value + 1))}>下一步</Button></div>
        </div>
      )}
    </div>
  );
}
