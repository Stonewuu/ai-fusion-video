"use client";

import { AlertCircle, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { MODEL_TYPE_LABELS } from "@/lib/api/ai-model";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  formatSemanticLabel,
  INHERIT_PROVIDER_PROTOCOL_VALUE,
  MODEL_PROTOCOL_LABELS,
} from "./model-config-support";

interface AiModelProtocolSectionProps {
  modelType: number;
  apiConfigName?: string;
  apiConfigDefaultProtocol: string;
  modelProtocolOverride: string;
  effectiveModelProtocol: string;
  protocolConfigured: boolean;
  protocolExpanded: boolean;
  onToggleExpand: () => void;
  options: Array<{ value: string; label: string }>;
  invalid: boolean;
  onSelectProtocol: (protocol: string) => void;
}

export function AiModelProtocolSection({
  modelType,
  apiConfigName,
  apiConfigDefaultProtocol,
  modelProtocolOverride,
  effectiveModelProtocol,
  protocolConfigured,
  protocolExpanded,
  onToggleExpand,
  options,
  invalid,
  onSelectProtocol,
}: AiModelProtocolSectionProps) {
  const showEditor =
    protocolExpanded || !protocolConfigured || Boolean(modelProtocolOverride);

  return (
    <div
      className={cn(
        "rounded-xl border px-3.5 py-2.5",
        !protocolConfigured
          ? "border-destructive/50 bg-destructive/5"
          : "border-border/30 bg-background/70",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs">
          <span className="font-medium text-foreground">请求协议</span>
          {effectiveModelProtocol ? (
            <Badge
              variant="secondary"
              className="bg-violet-500/10 text-violet-700 dark:text-violet-400"
            >
              {formatSemanticLabel(effectiveModelProtocol, MODEL_PROTOCOL_LABELS)}
            </Badge>
          ) : (
            <Badge variant="secondary" className="bg-destructive/10 text-destructive">
              <AlertCircle className="size-3" />
              未配置
            </Badge>
          )}
          <span className="text-muted-foreground">
            {modelProtocolOverride
              ? "模型单独覆盖"
              : apiConfigDefaultProtocol
                ? `继承自「${apiConfigName || "API 配置"}」`
                : "请补充协议"}
          </span>
        </div>

        {protocolConfigured && !modelProtocolOverride && (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={onToggleExpand}
          >
            {showEditor ? "收起" : "修改"}
            <ChevronRight
              className={cn(
                "transition-transform duration-200 motion-reduce:transition-none",
                showEditor && "rotate-90",
              )}
            />
          </Button>
        )}
      </div>

      {showEditor && (
        <div className="mt-3 flex flex-col gap-2 border-t border-border/20 pt-3">
          <Label htmlFor="model-protocol" className="text-xs text-muted-foreground">
            协议选择
          </Label>
          <Select
            value={modelProtocolOverride || INHERIT_PROVIDER_PROTOCOL_VALUE}
            onValueChange={(value) => {
              const nextProtocol =
                value === INHERIT_PROVIDER_PROTOCOL_VALUE ? "" : String(value);
              onSelectProtocol(nextProtocol);
            }}
            items={options}
          >
            <SelectTrigger
              id="model-protocol"
              aria-invalid={invalid}
              className="w-full text-sm"
            >
              <SelectValue
                placeholder={options[0]?.label || "继承 API 配置默认"}
              />
            </SelectTrigger>
            <SelectContent className="text-sm">
              <SelectGroup>
                {options.map((option) => (
                  <SelectItem
                    key={option.value}
                    value={option.value}
                    className="text-sm"
                  >
                    {option.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          {!protocolConfigured && (
            <p className="text-xs text-destructive">
              当前模型和所属 API 配置都未设置 {MODEL_TYPE_LABELS[modelType]} 请求协议，保存前必须至少配置一处。
            </p>
          )}
        </div>
      )}
    </div>
  );
}
