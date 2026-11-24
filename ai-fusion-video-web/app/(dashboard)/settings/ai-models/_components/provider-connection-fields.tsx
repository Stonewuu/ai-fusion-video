"use client";

import { useId, useState } from "react";
import { ChevronRight, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <>
      <div className="flex min-w-0 flex-col gap-2">
        <Label htmlFor={`${prefix}-name`} className="text-xs text-muted-foreground">
          配置名称
        </Label>
        <Input
          id={`${prefix}-name`}
          value={configuration.name}
          disabled={disabled}
          aria-invalid={invalidField === "name"}
          onChange={(event) => onChange({ name: event.target.value })}
        />
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <Label htmlFor={`${prefix}-key`} className="text-xs text-muted-foreground">
          API Key
        </Label>
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

      <Collapsible
        open={showAdvanced}
        onOpenChange={setShowAdvanced}
        className="rounded-xl border border-border/30 bg-background/70"
      >
        <CollapsibleTrigger className="flex w-full items-center justify-between gap-2 rounded-xl px-3.5 py-2.5 text-left text-xs font-medium text-muted-foreground outline-none transition-colors hover:bg-muted/35 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring">
          <span>高级设置（自定义接口地址）</span>
          <ChevronRight
            className={cn(
              "size-4 shrink-0 transition-transform duration-200 motion-reduce:transition-none",
              showAdvanced && "rotate-90",
            )}
          />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="flex min-w-0 flex-col gap-2 border-t border-border/20 p-3.5">
            <Label htmlFor={`${prefix}-url`} className="text-xs text-muted-foreground">
              API 地址
            </Label>
            <Input
              id={`${prefix}-url`}
              value={configuration.apiUrl ?? ""}
              disabled={disabled}
              onChange={(event) => onChange({ apiUrl: event.target.value })}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>
    </>
  );
}
