"use client";

import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { CanvasSelect } from "@/components/canvas/forms/canvas-select";
import { CanvasSegmentedControl } from "@/components/canvas/workspace/segmented-control";
import { canvasMcpApi, mcpEffectNames, type McpContract, type McpWorkflowTool, type McpEffect, type McpIdempotency, type McpOutputMode, type McpJsonSchema } from "@/lib/api/canvas-mcp";

export function McpContractEditor({ tools, contract, onSaved, onCancel }: {
  tools: McpWorkflowTool[]; contract: McpContract | null; onSaved: (value: McpContract) => void; onCancel: () => void;
}) {
  const id = useId();
  const [toolName, setToolName] = useState(contract?.definition.tool.toolName ?? "");
  const [name, setName] = useState(contract?.name ?? "");
  const [effect, setEffect] = useState<McpEffect>(contract?.definition.effect ?? "DESTRUCTIVE");
  const [idempotency, setIdempotency] = useState<McpIdempotency>(contract?.definition.idempotency ?? "NO_REPLAY");
  const [parameter, setParameter] = useState(contract?.definition.idempotencyParameter ?? "");
  const [outputMode, setOutputMode] = useState<McpOutputMode>(contract?.definition.outputMode ?? "TEXT");
  const [schema, setSchema] = useState(JSON.stringify(contract?.definition.outputSchema ?? { type: "string" }, null, 2));
  const [timeout, setTimeout] = useState(String(contract?.definition.timeoutSeconds ?? 60));
  const [enabled, setEnabled] = useState(contract?.enabled ?? true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const tool = tools.find((item) => item.toolName === toolName);
  const stringParameters = Object.entries(tool?.inputSchema.properties ?? {}).filter(([, value]) => value.type === "string");
  const selectTool = (value: string) => {
    const selected = tools.find((item) => item.toolName === value)!;
    setToolName(value); setName(selected.toolName);
    setEffect(selected.readOnly ? "READ_ONLY" : selected.destructive ? "DESTRUCTIVE" : "WRITE");
    setIdempotency(selected.readOnly ? "READ_ONLY" : "NO_REPLAY"); setParameter("");
    setOutputMode(selected.outputSchema ? "STRUCTURED_JSON" : "TEXT");
    setSchema(JSON.stringify(selected.outputSchema ?? { type: "string" }, null, 2));
  };
  const save = async () => {
    if (!tool || busy) return;
    setError(null);
    let outputSchema: McpJsonSchema;
    try {
      outputSchema = JSON.parse(schema);
      if (!outputSchema || typeof outputSchema !== "object" || Array.isArray(outputSchema)) throw new Error("请填写有效的结果格式对象");
      if (!Number.isInteger(Number(timeout)) || Number(timeout) < 1 || Number(timeout) > 600) throw new Error("超时需为 1–600 秒的整数");
    } catch (failure) { setError(failure instanceof Error ? failure.message : "结果格式无效"); return; }
    setBusy(true);
    try {
      onSaved(await canvasMcpApi.save({ id: contract?.id, expectedVersion: contract?.version, name: name.trim(), toolName,
        schemaHash: tool.schemaHash, configurationHash: tool.configurationHash, effect, idempotency,
        idempotencyParameter: idempotency === "KEYED" ? parameter : "", outputMode, outputSchema, timeoutSeconds: Number(timeout), enabled }));
    } catch (failure) { setError(failure instanceof Error ? failure.message : "操作保存失败"); }
    finally { setBusy(false); }
  };
  return (
    <section className="flex flex-col gap-4" aria-label={contract ? "编辑 MCP 画布操作" : "登记 MCP 画布操作"}>
      <div className="flex flex-col gap-2">
        <span className="text-sm font-medium">选择工具</span>
        <CanvasSelect label="选择 MCP 工具" value={toolName || null} items={tools.map((item) => ({ value: item.toolName, label: `${item.serverName} · ${item.toolName}` }))}
          placeholder="选择已连接的工具" onChange={selectTool} disabled={busy} />
        {tool?.description ? <p className="text-xs text-muted-foreground whitespace-pre-wrap">{tool.description}</p> : null}
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor={`${id}-name`} className="text-sm font-medium">操作名称</label>
        <Input id={`${id}-name`} value={name} maxLength={128} disabled={busy} onChange={(event) => setName(event.target.value)} placeholder="例如：查询素材资料" />
      </div>
      {tool ? (
        <>
          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">对外部内容的影响</span>
            {tool.readOnly || tool.destructive ? <p className="text-sm text-muted-foreground">{mcpEffectNames[effect]}</p>
              : <CanvasSegmentedControl label="MCP 操作影响" value={effect}
                options={[{ value: "WRITE", label: "修改外部内容", disabled: busy }, { value: "DESTRUCTIVE", label: "删除或覆盖", disabled: busy }]}
                onChange={(value) => { setEffect(value as McpEffect); setIdempotency("NO_REPLAY"); }} />}
          </div>
          {effect !== "READ_ONLY" ? (
            <div className="flex flex-col gap-2">
              <span className="text-sm font-medium">重复请求约定</span>
              <CanvasSegmentedControl label="MCP 幂等方式" value={idempotency} onChange={(value) => setIdempotency(value as McpIdempotency)}
                options={[{ value: "NO_REPLAY", label: "不重放", disabled: busy }, { value: "KEYED", label: "提供幂等键", disabled: busy || !stringParameters.length }]} />
              {idempotency === "KEYED" ? <CanvasSelect label="MCP 幂等键参数" value={parameter || null} disabled={busy}
                placeholder="选择服务用于去重的参数" items={stringParameters.map(([value, field]) => ({ value, label: field.title ?? value }))} onChange={setParameter} /> : null}
              <p className="text-xs text-muted-foreground">请求状态不明时需要核实原调用；即使提供幂等键，也不会自动重发。</p>
            </div>
          ) : null}
          <div className="flex flex-col gap-2">
            <span className="text-sm font-medium">返回内容</span>
            <CanvasSegmentedControl label="MCP 返回内容" value={outputMode}
              options={[{ value: "TEXT", label: "纯文本", disabled: busy }, { value: "JSON_TEXT", label: "JSON 文本", disabled: busy }, ...(tool.outputSchema ? [{ value: "STRUCTURED_JSON", label: "结构化 JSON", disabled: busy }] : [])]}
              onChange={(value) => { setOutputMode(value as McpOutputMode); setSchema(JSON.stringify(value === "TEXT" ? { type: "string" } : value === "STRUCTURED_JSON" ? tool.outputSchema : { type: "object" }, null, 2)); }} />
            <p className="text-xs text-muted-foreground">检查通过的内容会成为节点的文本输出，可连接后续智能体、生成或写入节点。</p>
          </div>
          <details className="rounded-lg border border-border/20 bg-background/70 p-3">
            <summary className="cursor-pointer text-sm">高级结果格式</summary>
            <div className="flex flex-col gap-2 pt-3">
              <label htmlFor={`${id}-schema`} className="text-sm font-medium">结果校验规则（JSON Schema）</label>
              <Textarea id={`${id}-schema`} value={schema} onChange={(event) => setSchema(event.target.value)} disabled={busy} rows={7} />
            </div>
          </details>
        </>
      ) : null}
      <div className="flex flex-col gap-2">
        <label htmlFor={`${id}-timeout`} className="text-sm font-medium">最长等待（秒）</label>
        <Input id={`${id}-timeout`} type="number" min={1} max={600} value={timeout} disabled={busy} onChange={(event) => setTimeout(event.target.value)} />
      </div>
      <label className="flex items-center gap-2 text-sm"><Checkbox checked={enabled} onCheckedChange={(value) => setEnabled(value === true)} disabled={busy} />允许新的画布调用</label>
      {error ? <p role="alert" className="text-sm text-destructive">{error}</p> : null}
      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={onCancel} disabled={busy}>返回</Button>
        <Button onClick={() => void save()} disabled={busy || !tool || !name.trim() || idempotency === "KEYED" && !parameter}>{busy ? "保存中…" : "保存操作"}</Button>
      </div>
    </section>
  );
}
