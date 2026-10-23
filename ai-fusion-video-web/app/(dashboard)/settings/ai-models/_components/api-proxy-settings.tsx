"use client";

import type { Dispatch, Ref, SetStateAction } from "react";
import { Eye, EyeOff } from "lucide-react";
import type { ApiConfigSaveReq } from "@/lib/api/ai-model";
import type { ConfigIssue } from "@/lib/settings/form-validation";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectGroup,
  SelectItem,
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
  detailsRef: Ref<HTMLDetailsElement>;
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
  detailsRef,
  updateField,
  showSecrets,
  setShowSecrets,
}: ApiProxySettingsProps) {
  return (
    <details
      ref={detailsRef}
      className="rounded-lg border border-border/20 bg-background/70 p-3"
    >
      <summary className="cursor-pointer rounded-lg text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring">
        出站代理 ·{" "}
        {form.proxyType && form.proxyType !== "none" ? "已启用" : "不使用"}
      </summary>
      <div className="mt-3 space-y-3">
        <div className="flex min-w-0 flex-col gap-2">
          <Label className="text-xs text-muted-foreground">出站代理</Label>
          <Select
            value={form.proxyType || "none"}
            onValueChange={(v) => {
              updateField("proxyType", v as string);
              if (v === "none") {
                updateField("proxyHost", "");
                updateField("proxyPort", undefined);
                updateField("proxyUsername", "");
                updateField("proxyPassword", "");
              }
            }}
            items={PROXY_TYPE_OPTIONS.map((o) => ({
              value: o.value,
              label: o.label,
            }))}
          >
            <SelectTrigger aria-label="出站代理" className="w-full text-sm">
              <SelectValue placeholder="选择代理类型" />
            </SelectTrigger>
            <SelectContent className="text-sm">
              <SelectGroup>
                {PROXY_TYPE_OPTIONS.map((opt) => (
                  <SelectItem
                    key={opt.value}
                    value={opt.value}
                    className="text-sm"
                  >
                    <div>
                      <div>{opt.label}</div>
                      <div className="text-[10px] text-muted-foreground">
                        {opt.description}
                      </div>
                    </div>
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          <p className="text-[10px] text-muted-foreground/70">
            仅影响后端访问当前 API 配置绑定的模型服务，适合 Vertex
            AI、OpenAI、Anthropic 等国外服务。支持可选代理账号密码。
          </p>
        </div>

        {form.proxyType && form.proxyType !== "none" && (
          <div className="space-y-3">
            <div className="grid grid-cols-[1fr_110px] gap-2">
              <div className="flex min-w-0 flex-col gap-2">
                <Label
                  htmlFor="api-proxy-host"
                  className="text-xs text-muted-foreground"
                >
                  代理主机
                </Label>
                <Input
                  placeholder="127.0.0.1"
                  id="api-proxy-host"
                  aria-invalid={issue?.field === "api-proxy-host"}
                  value={form.proxyHost || ""}
                  onChange={(e) => updateField("proxyHost", e.target.value)}
                  className="text-sm font-mono"
                />
              </div>
              <div className="flex min-w-0 flex-col gap-2">
                <Label
                  htmlFor="api-proxy-port"
                  className="text-xs text-muted-foreground"
                >
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
                  onChange={(e) =>
                    updateField(
                      "proxyPort",
                      e.target.value ? Number(e.target.value) : undefined,
                    )
                  }
                  className="text-sm font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="flex min-w-0 flex-col gap-2">
                <Label
                  htmlFor="api-proxy-user"
                  className="text-xs text-muted-foreground"
                >
                  代理用户名
                </Label>
                <Input
                  placeholder="可选"
                  id="api-proxy-user"
                  aria-invalid={issue?.field === "api-proxy-user"}
                  value={form.proxyUsername || ""}
                  onChange={(e) => updateField("proxyUsername", e.target.value)}
                  className="text-sm"
                />
              </div>
              <div className="flex min-w-0 flex-col gap-2">
                <Label
                  htmlFor="api-proxy-password"
                  className="text-xs text-muted-foreground"
                >
                  代理密码
                </Label>
                <div className="relative">
                  <Input
                    id="api-proxy-password"
                    type={showSecrets.proxyPassword ? "text" : "password"}
                    placeholder="可选"
                    value={form.proxyPassword || ""}
                    onChange={(e) =>
                      updateField("proxyPassword", e.target.value)
                    }
                    className="text-sm pr-9"
                  />
                  <button
                    type="button"
                    aria-label={
                      showSecrets.proxyPassword
                        ? "隐藏代理密码"
                        : "显示代理密码"
                    }
                    onClick={() =>
                      setShowSecrets((prev) => ({
                        ...prev,
                        proxyPassword: !prev.proxyPassword,
                      }))
                    }
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showSecrets.proxyPassword ? (
                      <EyeOff className="h-3.5 w-3.5" />
                    ) : (
                      <Eye className="h-3.5 w-3.5" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </details>
  );
}
