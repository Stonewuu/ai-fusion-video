"use client";

import { useId, useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ApiConfigSaveReq } from "@/lib/api/ai-model";

interface ProviderConnectionFieldsProps {
  configuration: ApiConfigSaveReq;
  disabled: boolean;
  invalidField: "name" | "key" | null;
  onChange: (patch: Partial<ApiConfigSaveReq>) => void;
}

export function ProviderConnectionFields({
  configuration,
  disabled,
  invalidField,
  onChange,
}: ProviderConnectionFieldsProps) {
  const prefix = useId();
  const [showKey, setShowKey] = useState(false);
  return (
    <>
      <div className="flex min-w-0 flex-col gap-2">
        <Label htmlFor={`${prefix}-name`}>配置名称</Label>
        <Input
          id={`${prefix}-name`}
          value={configuration.name}
          disabled={disabled}
          aria-invalid={invalidField === "name"}
          onChange={(event) => onChange({ name: event.target.value })}
        />
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <Label htmlFor={`${prefix}-key`}>API Key</Label>
        <div className="flex min-w-0 items-center gap-2">
          <Input
            id={`${prefix}-key`}
            type={showKey ? "text" : "password"}
            value={configuration.apiKey ?? ""}
            disabled={disabled}
            autoComplete="off"
            placeholder="粘贴平台 API Key"
            aria-invalid={invalidField === "key"}
            onChange={(event) => onChange({ apiKey: event.target.value })}
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            disabled={disabled}
            aria-label={showKey ? "隐藏密钥" : "显示密钥"}
            onClick={() => setShowKey((value) => !value)}
          >
            {showKey ? <EyeOff /> : <Eye />}
          </Button>
        </div>
      </div>
      <details>
        <summary className="cursor-pointer rounded-sm text-xs text-muted-foreground focus-visible:outline-2 focus-visible:outline-offset-2">
          高级设置
        </summary>
        <div className="mt-3 flex min-w-0 flex-col gap-2 p-1">
          <Label htmlFor={`${prefix}-url`}>API 地址</Label>
          <Input
            id={`${prefix}-url`}
            value={configuration.apiUrl ?? ""}
            disabled={disabled}
            onChange={(event) => onChange({ apiUrl: event.target.value })}
          />
        </div>
      </details>
    </>
  );
}
