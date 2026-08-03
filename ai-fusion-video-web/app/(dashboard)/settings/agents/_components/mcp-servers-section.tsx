"use client";

import { useConfirm } from "@/components/ui/confirm-dialog";
import { useState } from "react";
import {
  Cable,
  CheckCircle2,
  Loader2,
  Pencil,
  Plus,
  Power,
  TestTube2,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { agentConfigApi, type AgentMcpServer } from "@/lib/api/agent-config";
import { settingsTypography } from "../../_shared";

const TRANSPORT_ITEMS = [
  { value: "http", label: "Streamable HTTP" },
  { value: "sse", label: "SSE" },
];

const STATUS_ITEMS = [
  { value: "1", label: "启用" },
  { value: "0", label: "停用" },
];

interface McpServersSectionProps {
  servers: AgentMcpServer[];
  onRefresh: () => Promise<void>;
}

interface McpForm {
  name: string;
  transport: "http" | "sse";
  url: string;
  headers: string;
  queryParams: string;
  enabledTools: string;
  timeoutSeconds: string;
  initializationTimeoutSeconds: string;
  status: string;
}

const EMPTY_FORM: McpForm = {
  name: "",
  transport: "http",
  url: "",
  headers: "",
  queryParams: "",
  enabledTools: "",
  timeoutSeconds: "120",
  initializationTimeoutSeconds: "30",
  status: "1",
};

function mapToLines(values: Record<string, string>) {
  return Object.entries(values).map(([key, value]) => `${key}: ${value}`).join("\n");
}

function linesToMap(value: string) {
  const result: Record<string, string> = {};
  for (const rawLine of value.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const separator = line.indexOf(":");
    if (separator <= 0) throw new Error(`格式错误：${line}`);
    result[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }
  return result;
}

export function McpServersSection({ servers, onRefresh }: McpServersSectionProps) {
  const { confirm } = useConfirm();
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<AgentMcpServer | null>(null);
  const [form, setForm] = useState<McpForm>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [testingId, setTestingId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const showEditor = (server?: AgentMcpServer) => {
    setEditing(server || null);
    setForm(server ? {
      name: server.name,
      transport: server.transport,
      url: server.url,
      headers: mapToLines(server.headers),
      queryParams: mapToLines(server.queryParams),
      enabledTools: server.enabledTools.join(", "),
      timeoutSeconds: String(server.timeoutSeconds),
      initializationTimeoutSeconds: String(server.initializationTimeoutSeconds),
      status: String(server.status),
    } : EMPTY_FORM);
    setOpen(true);
  };

  const save = async () => {
    if (!form.name.trim() || !form.url.trim()) {
      toast.error("请填写 MCP 名称和服务地址");
      return;
    }
    const timeoutSeconds = Number(form.timeoutSeconds);
    const initializationTimeoutSeconds = Number(form.initializationTimeoutSeconds);
    if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 1 || timeoutSeconds > 600
      || !Number.isInteger(initializationTimeoutSeconds)
      || initializationTimeoutSeconds < 1
      || initializationTimeoutSeconds > 600) {
      toast.error("超时时间需为 1 到 600 秒之间的整数");
      return;
    }
    setSaving(true);
    try {
      await agentConfigApi.saveMcpServer({
        id: editing?.id,
        name: form.name,
        transport: form.transport,
        url: form.url,
        headers: linesToMap(form.headers),
        queryParams: linesToMap(form.queryParams),
        enabledTools: form.enabledTools.split(",").map((value) => value.trim()).filter(Boolean),
        protocolVersions: [],
        timeoutSeconds,
        initializationTimeoutSeconds,
        status: Number(form.status),
      });
      toast.success(editing ? "MCP 服务已更新" : "MCP 服务已创建");
      setOpen(false);
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "保存 MCP 服务失败");
    } finally {
      setSaving(false);
    }
  };

  const test = async (server: AgentMcpServer) => {
    setTestingId(server.id);
    try {
      const result = await agentConfigApi.testMcpServer(server.id);
      if (result.success) toast.success(result.message);
      else toast.error(result.message);
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "MCP 连接测试失败");
    } finally {
      setTestingId(null);
    }
  };

  const remove = async (server: AgentMcpServer) => {
    const ok = await confirm({ title: "删除 MCP 服务", description: `确认删除 MCP 服务「${server.name}」？`, variant: "destructive", confirmText: "确定删除" }); if (!ok) return;
    setDeletingId(server.id);
    try {
      await agentConfigApi.deleteMcpServer(server.id);
      toast.success("MCP 服务已删除");
      await onRefresh();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "删除 MCP 服务失败");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <section className="flex flex-col rounded-xl border border-border/30 bg-card/50 p-4 backdrop-blur-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="rounded-xl bg-primary/10 p-2 text-primary">
            <Cable className="size-5" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className={settingsTypography.sectionTitle}>我的 MCP</h2>
              <Badge variant="secondary">{servers.length}</Badge>
            </div>
            <p className={settingsTypography.sectionDescription}>连接远程 HTTP/SSE 服务，发现的工具可通过 / 加入当前对话。</p>
          </div>
        </div>
        <Button onClick={() => showEditor()}><Plus />添加 MCP</Button>
      </div>

      <div className="mt-4 flex-1 overflow-hidden rounded-lg border border-border/20 bg-background/70">
        {servers.length === 0 ? (
          <div className="flex min-h-32 flex-col items-center justify-center p-6 text-center">
            <Cable className="size-5 text-muted-foreground" />
            <p className="mt-2 text-sm font-medium">还没有自定义 MCP 服务</p>
            <p className="mt-1 text-xs text-muted-foreground">添加公网服务后，助手会发现并按权限调用其中的工具。</p>
          </div>
        ) : servers.map((server) => (
          <div
            key={server.id}
            className="flex items-center justify-between gap-3 border-b border-border/20 p-3 transition-colors last:border-b-0 hover:bg-muted/40"
          >
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="truncate text-sm font-medium">{server.name}</span>
                <Badge variant="outline">{server.transport.toUpperCase()}</Badge>
                <Badge variant={server.status === 1 ? "secondary" : "outline"}>
                  {server.status === 1 ? "已启用" : "已停用"}
                </Badge>
                <Badge variant="outline">
                  {server.enabledTools.length > 0 ? `${server.enabledTools.length} 个工具` : "全部工具"}
                </Badge>
              </div>
              <div className="mt-1 truncate text-xs text-muted-foreground">{server.url}</div>
              {server.lastTestMessage && (
                <div className={server.lastTestStatus === "success"
                  ? "mt-1 flex items-center gap-1 text-xs text-muted-foreground"
                  : "mt-1 text-xs text-destructive"}
                >
                  {server.lastTestStatus === "success" && <CheckCircle2 className="size-3.5" />}
                  <span className="truncate">{server.lastTestMessage}</span>
                </div>
              )}
            </div>
            <div className="flex shrink-0 gap-2">
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`测试 ${server.name}`}
                title="测试连接"
                onClick={() => test(server)}
                disabled={testingId === server.id}
              >
                {testingId === server.id ? <Loader2 className="animate-spin" /> : <TestTube2 />}
              </Button>
              <Button variant="ghost" size="icon-sm" aria-label={`编辑 ${server.name}`} title="编辑" onClick={() => showEditor(server)}>
                <Pencil />
              </Button>
              <Button
                variant="destructive-ghost"
                size="icon-sm"
                aria-label={`删除 ${server.name}`}
                title="删除"
                onClick={() => remove(server)}
                disabled={deletingId === server.id}
              >
                {deletingId === server.id ? <Loader2 className="animate-spin" /> : <Trash2 />}
              </Button>
            </div>
          </div>
        ))}
      </div>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden rounded-2xl sm:max-w-2xl">
          <DialogHeader className="shrink-0 pr-8">
            <DialogTitle>{editing ? "编辑 MCP 服务" : "添加 MCP 服务"}</DialogTitle>
            <DialogDescription>用户配置支持远程 HTTP/SSE；本机命令型 stdio MCP 由管理员在服务端配置。</DialogDescription>
          </DialogHeader>

          <div className="-mx-1 min-h-0 overflow-y-auto px-1 py-1">
            <div className="space-y-6">
              <section aria-labelledby="mcp-connection-heading" className="space-y-3">
                <div>
                  <h3 id="mcp-connection-heading" className={settingsTypography.sectionTitle}>连接信息</h3>
                  <p className="mt-1 text-xs text-muted-foreground">填写服务标识、传输协议和可公开访问的端点。</p>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label htmlFor="mcp-name" className={settingsTypography.fieldLabel}>名称</Label>
                    <Input id="mcp-name" value={form.name} onChange={(event) => setForm((value) => ({ ...value, name: event.target.value }))} placeholder="notion-tools" />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="mcp-transport" className={settingsTypography.fieldLabel}>传输方式</Label>
                    <Select items={TRANSPORT_ITEMS} value={form.transport} onValueChange={(value) => value && setForm((formValue) => ({ ...formValue, transport: value as "http" | "sse" }))}>
                      <SelectTrigger id="mcp-transport" className="w-full"><SelectValue placeholder="选择传输方式" /></SelectTrigger>
                      <SelectContent><SelectGroup>{TRANSPORT_ITEMS.map((item) => <SelectItem key={item.value} value={item.value}>{item.label}</SelectItem>)}</SelectGroup></SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5 sm:col-span-2">
                    <Label htmlFor="mcp-url" className={settingsTypography.fieldLabel}>服务地址</Label>
                    <Input id="mcp-url" value={form.url} onChange={(event) => setForm((value) => ({ ...value, url: event.target.value }))} placeholder="https://mcp.example.com/mcp" />
                  </div>
                </div>
              </section>

              <section aria-labelledby="mcp-auth-heading" className="space-y-3 border-t border-border/20 pt-5">
                <div>
                  <h3 id="mcp-auth-heading" className={settingsTypography.sectionTitle}>鉴权与参数</h3>
                  <p className="mt-1 text-xs text-muted-foreground">每行填写一个 key: value；不需要时保持为空。</p>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5">
                    <Label htmlFor="mcp-headers" className={settingsTypography.fieldLabel}>请求头</Label>
                    <Textarea id="mcp-headers" value={form.headers} onChange={(event) => setForm((value) => ({ ...value, headers: event.target.value }))} placeholder="Authorization: Bearer ..." className="min-h-24 font-mono text-sm" />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="mcp-query" className={settingsTypography.fieldLabel}>查询参数</Label>
                    <Textarea id="mcp-query" value={form.queryParams} onChange={(event) => setForm((value) => ({ ...value, queryParams: event.target.value }))} placeholder="tenant: demo" className="min-h-24 font-mono text-sm" />
                  </div>
                </div>
              </section>

              <section aria-labelledby="mcp-runtime-heading" className="space-y-3 border-t border-border/20 pt-5">
                <div>
                  <h3 id="mcp-runtime-heading" className={settingsTypography.sectionTitle}>运行设置</h3>
                  <p className="mt-1 text-xs text-muted-foreground">限制可用工具，并为初始化和单次调用设置超时。</p>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-1.5 sm:col-span-2">
                    <Label htmlFor="mcp-tools" className={settingsTypography.fieldLabel}>启用工具</Label>
                    <Input id="mcp-tools" value={form.enabledTools} onChange={(event) => setForm((value) => ({ ...value, enabledTools: event.target.value }))} placeholder="留空启用全部；多个工具用逗号分隔" />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="mcp-init-timeout" className={settingsTypography.fieldLabel}>初始化超时（秒）</Label>
                    <Input id="mcp-init-timeout" type="number" min={1} max={600} value={form.initializationTimeoutSeconds} onChange={(event) => setForm((value) => ({ ...value, initializationTimeoutSeconds: event.target.value }))} />
                  </div>
                  <div className="space-y-1.5">
                    <Label htmlFor="mcp-timeout" className={settingsTypography.fieldLabel}>调用超时（秒）</Label>
                    <Input id="mcp-timeout" type="number" min={1} max={600} value={form.timeoutSeconds} onChange={(event) => setForm((value) => ({ ...value, timeoutSeconds: event.target.value }))} />
                  </div>
                  <div className="space-y-1.5 sm:col-span-2">
                    <Label htmlFor="mcp-status" className={settingsTypography.fieldLabel}>状态</Label>
                    <Select items={STATUS_ITEMS} value={form.status} onValueChange={(value) => value && setForm((formValue) => ({ ...formValue, status: value }))}>
                      <SelectTrigger id="mcp-status" className="w-full"><SelectValue placeholder="选择状态" /></SelectTrigger>
                      <SelectContent><SelectGroup>
                        <SelectItem value="1"><Power />启用</SelectItem>
                        <SelectItem value="0">停用</SelectItem>
                      </SelectGroup></SelectContent>
                    </Select>
                  </div>
                </div>
              </section>
            </div>
          </div>

          <DialogFooter className="shrink-0 border-t border-border/20 pt-4">
            <Button variant="outline" onClick={() => setOpen(false)}>取消</Button>
            <Button onClick={save} disabled={saving}>
              {saving && <Loader2 className="animate-spin" />}
              保存
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
