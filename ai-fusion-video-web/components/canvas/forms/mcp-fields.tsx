"use client";

import { useEffect, useId, useState } from "react";
import { Cable, Settings2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useAuthStore } from "@/lib/store/auth-store";
import { canvasMcpApi, mcpEffectNames, type McpContract } from "@/lib/api/canvas-mcp";
import { McpContractsDialog } from "@/components/mcp/contracts-dialog";
import { McpArgumentFields } from "@/components/mcp/argument-fields";
import type { CanvasConfig } from "@/lib/canvas/types";
import { CanvasSelect } from "./canvas-select";
import { CanvasSegmentedControl } from "../workspace/segmented-control";

export function CanvasMcpFields({ config, onChange, connectedPorts, disabled }: {
  config: CanvasConfig; onChange: (next: CanvasConfig) => void; connectedPorts: string[]; disabled: boolean;
}) {
  const userId = useAuthStore((state) => state.user?.id);
  const id = useId();
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState<{ key: string; items: McpContract[]; error: string | null } | null>(null);
  const [manage, setManage] = useState(false);
  const key = `${userId}:${attempt}`;
  const data = loaded?.key === key ? loaded : null;
  const contracts = data?.items.filter((item) => item.enabled) ?? [];
  const selected = contracts.find((item) => item.id === config.contractId);
  const mode = String(config.inputMode ?? "NONE");
  const connected = connectedPorts.includes("input");
  useEffect(() => {
    if (!userId) return;
    let active = true;
    void canvasMcpApi.list().then((items) => { if (active) setLoaded({ key, items, error: null }); })
      .catch((failure: unknown) => { if (active) setLoaded({ key, items: [], error: failure instanceof Error ? failure.message : "操作目录加载失败" }); });
    return () => { active = false; };
  }, [key, userId]);
  const update = (field: string, value: string | number) => onChange({ ...config, [field]: value });
  return (
    <section className="flex flex-col gap-4" aria-label="MCP 工具设置">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-sm font-medium"><Cable className="size-4 text-primary" />我的画布操作</span>
        <div className="flex items-center gap-2"><Button variant="ghost" size="icon" aria-label="刷新 MCP 操作" disabled={disabled || !data} onClick={() => setAttempt((value) => value + 1)}><RefreshCw /></Button>
          <Button variant="outline" size="default" disabled={disabled} onClick={() => setManage(true)}><Settings2 />管理操作</Button></div>
      </div>
      <CanvasSelect label="画布 MCP 操作" value={selected?.id ?? null} disabled={disabled || !data} placeholder={!data ? "正在加载操作…" : "选择已登记的操作"}
        items={contracts.map((item) => ({ value: item.id, label: item.name }))} onChange={(value) => {
          const item = contracts.find((contract) => contract.id === value)!;
          const parameters = Object.entries(item.definition.tool.inputSchema.properties ?? {})
            .filter(([name, field]) => field.type === "string" && name !== item.definition.idempotencyParameter);
          onChange({ ...config, contractId: item.id, contractVersion: item.version, argumentsJson: "{}",
            inputMode: connected ? parameters.length ? "TEXT_FIELD" : "JSON_OBJECT" : "NONE",
            inputParameter: parameters.length === 1 ? parameters[0][0] : "" });
        }} />
      {data?.error ? <p role="alert" className="text-sm text-destructive">{data.error}</p> : null}
      {data && !contracts.length && !data.error ? <p className="text-sm text-muted-foreground">先登记一个 MCP 画布操作，再选择工具和填写参数。</p> : null}
      {data && config.contractId && !selected ? <p className="text-sm text-destructive">原操作已停用、移除或不属于当前账号，请重新选择。</p> : null}
      {selected ? <>
        <p className="text-xs text-muted-foreground">{selected.definition.tool.serverName} · {mcpEffectNames[selected.definition.effect]} · 最长等待 {selected.definition.timeoutSeconds} 秒</p>
        {config.contractVersion !== selected.version ? <div className="flex flex-col gap-2 rounded-lg border border-border/30 bg-background/70 p-3">
          <p className="text-sm">此操作有新版本，请确认后检查参数。</p><Button variant="outline" size="sm" disabled={disabled} onClick={() => update("contractVersion", selected.version)}>采用当前版本</Button>
        </div> : null}
        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">前置步骤内容如何传入</span>
          <CanvasSegmentedControl label="MCP 输入方式" value={mode}
            options={[{ value: "NONE", label: "固定参数", disabled }, { value: "TEXT_FIELD", label: "接收文本", disabled }, { value: "JSON_OBJECT", label: "接收 JSON", disabled }]}
            onChange={(value) => {
              const next: CanvasConfig = { ...config, inputMode: value, ...(value === "JSON_OBJECT" ? { argumentsJson: "{}" } : {}) };
              if (value === "NONE") delete next.input;
              onChange(next);
            }} />
        </div>
        {mode === "TEXT_FIELD" ? <CanvasSelect label="接收前置步骤文本的参数" value={typeof config.inputParameter === "string" && config.inputParameter ? config.inputParameter : null}
          placeholder="选择接收文本的参数" disabled={disabled}
          items={Object.entries(selected.definition.tool.inputSchema.properties ?? {}).filter(([name, field]) => field.type === "string" && name !== selected.definition.idempotencyParameter)
            .map(([value, field]) => ({ value, label: field.title ?? value }))} onChange={(value) => update("inputParameter", value)} /> : null}
        {mode !== "NONE" && !connected ? <div className="flex flex-col gap-2"><label htmlFor={`${id}-input`} className="text-sm font-medium">输入内容</label>
          <Textarea id={`${id}-input`} rows={4} value={typeof config.input === "string" ? config.input : ""} disabled={disabled} onChange={(event) => update("input", event.target.value)} />
          <p className="text-xs text-muted-foreground">也可以连接前置步骤节点的文本输出。</p></div> : null}
        {mode === "NONE" && connected ? <p className="text-sm text-destructive">已有前置步骤连接，请选择一种接收方式。</p> : null}
        {mode !== "JSON_OBJECT" ? <McpArgumentFields schema={selected.definition.tool.inputSchema} value={String(config.argumentsJson ?? "{}")}
          disabled={disabled} onChange={(value) => update("argumentsJson", value)} omitted={[selected.definition.idempotencyParameter, ...(mode === "TEXT_FIELD" ? [String(config.inputParameter ?? "")] : [])].filter(Boolean)} /> : null}
      </> : null}
      {manage ? <McpContractsDialog key={userId} onClose={() => setManage(false)} onChanged={() => setAttempt((value) => value + 1)} /> : null}
    </section>
  );
}
