"use client";

import type { Dispatch, SetStateAction } from "react";
import { ChevronRight, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ApiConfigSaveReq } from "@/lib/api/ai-model";
import type { ConfigIssue } from "@/lib/settings/form-validation";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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

const PROXY_TYPE_OPTIONS = [
  { value: "none", label: "不使用代理", description: "直连模型服务" },
  {
    value: "socks5",
    label: "SOCKS5",
    description: "常见本地代理，如 127.0.0.1:7890",
  },
  { value: "http", label: "HTTP", description: "HTTP/HTTPS 出站代理" },
] as const;

interface ApiProxySettingsProps {
  form: ApiConfigSaveReq;
  issue: ConfigIssue | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  updateField: <K extends keyof ApiConfigSaveReq>(
    key: K,
    value: ApiConfigSaveReq[K],
  ) => void;
  showSecrets: Record<string, boolean>;
  setShowSecrets: Dispatch<SetStateAction<Record<string, boolean>>>;
}

export function ApiProxySettings({
  form,
  issue,
  open,
  onOpenChange,
  updateField,
  showSecrets,
  setShowSecrets,
}: ApiProxySettingsProps) {
  const proxyEnabled = Boolean(form.proxyType && form.proxyType !== "none");
  const proxyLabel = proxyEnabled
    ? PROXY_TYPE_OPTIONS.find((item) => item.value === form.proxyType)?.label ?? form.proxyType
    : "未启用";

  return (
    <Collapsible
      open={open}
      onOpenChange={onOpenChange}
      className="rounded-xl border border-border/30 bg-background/70"
    >
      <CollapsibleTrigger className="flex w-full items-center justify-between gap-3 rounded-xl px-3.5 py-2.5 text-left outline-none transition-colors hover:bg-muted/35 focus-visible:ring-2 focus-visible:ring-ring">
        <span className="flex min-w-0 items-center gap-2">
          <span className="text-sm font-medium text-foreground">出站代理</span>
          <Badge variant={proxyEnabled ? "secondary" : "outline"} className="text-xs">
            {proxyLabel}
          </Badge>
        </span>
        <ChevronRight
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform duration-200 motion-reduce:transition-none",
            open && "rotate-90",
          )}
        />
      </CollapsibleTrigger>

      <CollapsibleContent>
        <div className="space-y-3 border-t border-border/20 p-3.5">
          <div className="flex min-w-0 flex-col gap-2">
            <Label className="text-xs text-muted-foreground">代理类型</Label>
            <Select
              value={form.proxyType || "none"}
              onValueChange={(value) => {
                updateField("proxyType", value as string);
                if (value === "none") {
                  updateField("proxyHost", "");
                  updateField("proxyPort", undefined);
                  updateField("proxyUsername", "");
                  updateField("proxyPassword", "");
                }
              }}
              items={PROXY_TYPE_OPTIONS.map((option) => ({
                value: option.value,
                label: option.label,
              }))}
            >
              <SelectTrigger aria-label="出站代理" className="w-full text-sm">
                <SelectValue placeholder="选择代理类型" />
              </SelectTrigger>
              <SelectContent className="text-sm">
                <SelectGroup>
                  {PROXY_TYPE_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value} className="text-sm">
                      <div>
                        <div>{opt.label}</div>
                        <div className="text-xs text-muted-foreground">
                          {opt.description}
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
            <p className="text-xs leading-relaxed text-muted-foreground">
              仅用于服务端访问当前配置绑定的模型接口，支持可选代理认证。
            </p>
          </div>

          {proxyEnabled && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_120px]">
                <div className="flex min-w-0 flex-col gap-2">
                  <Label htmlFor="api-proxy-host" className="text-xs text-muted-foreground">
                    代理主机
                  </Label>
                  <Input
                    placeholder="127.0.0.1"
                    id="api-proxy-host"
                    aria-invalid={issue?.field === "api-proxy-host"}
                    value={form.proxyHost || ""}
                    onChange={(event) => updateField("proxyHost", event.target.value)}
                    className="font-mono text-sm"
                  />
                </div>
                <div className="flex min-w-0 flex-col gap-2">
                  <Label htmlFor="api-proxy-port" className="text-xs text-muted-foreground">
                    端口
                  </Label>
                  <Input
                    type="number"
                    min={1}
                    max={65535}
                    placeholder="7890"
                    id="api-proxy-port"
                    aria-invalid={issue?.field === "api-proxy-port"}
                    value={form.proxyPort ?? ""}
                    onChange={(event) =>
                      updateField(
                        "proxyPort",
                        event.target.value ? Number(event.target.value) : undefined,
                      )
                    }
                    className="font-mono text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div className="flex min-w-0 flex-col gap-2">
                  <Label htmlFor="api-proxy-user" className="text-xs text-muted-foreground">
                    代理用户名（可选）
                  </Label>
                  <Input
                    placeholder="可选"
                    id="api-proxy-user"
                    aria-invalid={issue?.field === "api-proxy-user"}
                    value={form.proxyUsername || ""}
                    onChange={(event) => updateField("proxyUsername", event.target.value)}
                    className="text-sm"
                  />
                </div>
                <div className="flex min-w-0 flex-col gap-2">
                  <Label htmlFor="api-proxy-password" className="text-xs text-muted-foreground">
                    代理密码（可选）
                  </Label>
                  <div className="relative">
                    <Input
                      id="api-proxy-password"
                      type={showSecrets.proxyPassword ? "text" : "password"}
                      placeholder="可选"
                      value={form.proxyPassword || ""}
                      onChange={(event) => updateField("proxyPassword", event.target.value)}
                      className="pr-9 text-sm"
                    />
                    <button
                      type="button"
                      aria-label={showSecrets.proxyPassword ? "隐藏代理密码" : "显示代理密码"}
                      aria-pressed={!!showSecrets.proxyPassword}
                      onClick={() =>
                        setShowSecrets((prev) => ({
                          ...prev,
                          proxyPassword: !prev.proxyPassword,
                        }))
                      }
                      className="absolute right-2.5 top-1/2 flex size-6 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {showSecrets.proxyPassword ? (
                        <EyeOff className="size-3.5" />
                      ) : (
                        <Eye className="size-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
