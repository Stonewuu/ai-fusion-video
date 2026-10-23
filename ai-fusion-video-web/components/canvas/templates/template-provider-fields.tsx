"use client";
import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { canvasApi } from "@/lib/api/canvas";
import { canvasMcpApi, type McpContract } from "@/lib/api/canvas-mcp";
import { getApiErrorMessage } from "@/lib/api/api-error";
import type { CanvasCatalog } from "@/lib/canvas/types";
import type { CanvasTemplate, TemplateBinding, TemplateBindings, TemplateProvider } from "@/lib/api/canvas-templates";
import { CanvasSelect } from "../forms/canvas-select";

const KINDS = { TEXT_MODEL: "文本模型", IMAGE_MODEL: "图片模型", VIDEO_MODEL: "视频模型", COMFYUI_MODEL: "工作流模型", AUDIO_API: "配乐服务", WORKFLOW: "工作流版本", MCP: "MCP 操作" };
export function TemplateProviderFields({ template, bindings, disabled, onChange }: {
  template: CanvasTemplate; bindings: TemplateBindings; disabled: boolean;
  onChange: (provider: TemplateProvider, binding: TemplateBinding) => void;
}) {
  const [attempt, setAttempt] = useState(0);
  const [data, setData] = useState<{ catalog: CanvasCatalog; contracts: McpContract[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void Promise.all([canvasApi.capabilities(), template.providers.some(item => item.kind === "MCP") ? canvasMcpApi.list() : Promise.resolve([])])
      .then(([catalog, contracts]) => { if (active) { setData({ catalog, contracts }); setError(null); } })
      .catch(failure => { if (active) setError(getApiErrorMessage(failure)); });
    return () => { active = false; };
  }, [template, attempt]);
  const items = (slot: TemplateProvider) => {
    if (!data) return [];
    if (slot.kind === "MCP") return data.contracts.filter(item => item.enabled).map(item => ({ value: item.id, label: `${item.name} · v${item.version}` }));
    if (slot.kind === "AUDIO_API") return data.catalog.audioServices.map(item => ({ value: String(item.id), label: item.name }));
    if (slot.kind === "WORKFLOW") {
      const selected = slot.uses.map(use => {
        const model = template.providers.find(provider => provider.kind === "COMFYUI_MODEL" && provider.uses.some(item => item.nodeId === use.nodeId && item.batch === use.batch));
        return model ? data.catalog.models.find(item => String(item.id) === bindings.providers[model.id]?.id) : undefined;
      });
      return selected[0]?.versions.filter(version => selected.every(model => model?.versions.some(item => item.id === version.id && !item.unavailable)))
        .map(version => ({ value: String(version.id), label: `${selected[0]!.name} · 版本 ${version.number}` })) ?? [];
    }
    const kind = { TEXT_MODEL: "text", IMAGE_MODEL: "image", VIDEO_MODEL: "video", COMFYUI_MODEL: "comfyui" }[slot.kind];
    return data.catalog.models.filter(item => item.kind === kind && !item.unavailable && (kind !== "comfyui" || slot.uses.every(use => {
      const node = template.graph.nodes.find(node => node.id === use.nodeId)!;
      const config = use.batch ? node.config.operationConfig as Record<string, unknown> : node.config;
      return config.resultType === item.resultType;
    }))).map(item => ({ value: String(item.id), label: item.name }));
  };
  return <section className="flex flex-col gap-4" aria-label="模板模型与工具">
    <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-medium">选择模型与工具</h3>
      <Button variant="ghost" size="icon-sm" aria-label="刷新模型和工具" disabled={disabled} onClick={() => setAttempt(value => value + 1)}><RefreshCw /></Button></div>
    {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
    {[...template.providers].sort((a, b) => Number(a.kind === "WORKFLOW") - Number(b.kind === "WORKFLOW")).map(slot => {
      const choices = items(slot);
      const binding = bindings.providers[slot.id];
      const currentVersion = slot.kind !== "MCP" || data?.contracts.find(item => item.id === binding?.id)?.version === binding?.version;
      return <div key={slot.id} className="flex flex-col gap-2">
        <p className="text-sm">{slot.label}</p>
        <CanvasSelect label={slot.label} value={currentVersion && choices.some(item => item.value === binding?.id) ? binding.id : null}
          items={choices} placeholder={!data ? "正在加载…" : `选择${KINDS[slot.kind]}`} disabled={disabled || !data}
          onChange={id => onChange(slot, slot.kind === "MCP" ? { id, version: data!.contracts.find(item => item.id === id)!.version } : { id })} />
        {data && !choices.length && <p className="text-xs text-muted-foreground">{slot.kind === "WORKFLOW" ? "先选择上方工作流模型，再选择该模型的已发布版本。" : "暂无可用选项，请配置对应服务后刷新。"}</p>}
        {data && binding && !currentVersion && <p className="text-xs text-destructive">操作版本已变化，请重新选择并检查参数。</p>}
      </div>;
    })}
  </section>;
}
