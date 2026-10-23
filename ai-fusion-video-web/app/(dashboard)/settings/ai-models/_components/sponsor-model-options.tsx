"use client";

import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

const options = [
  {
    capability: "supportContextIr",
    key: "contextIrEnabled",
    label: "Context IR",
    description: "由平台优化创作提示词。",
  },
  {
    capability: "supportPriorityChannel",
    key: "fastMode",
    label: "优先通道",
    description: "使用平台优先队列，按平台对应费率计费。",
  },
];

export function SponsorModelOptions({
  config,
  onChange,
}: {
  config: Record<string, unknown>;
  onChange: (key: string, enabled: boolean) => void;
}) {
  const available = options.filter(
    (option) => config[option.capability] === true,
  );
  if (!available.length) {
    return null;
  }
  return (
    <div className="flex flex-col gap-4 rounded-lg border border-border/20 bg-background/70 p-4">
      {available.map((option) => (
        <div
          key={option.key}
          className="flex min-w-0 items-center justify-between gap-4"
        >
          <div className="flex min-w-0 flex-col gap-1">
            <Label htmlFor={`sponsor-${option.key}`}>{option.label}</Label>
            <p className="text-xs text-muted-foreground">
              {option.description}
            </p>
          </div>
          <Checkbox
            id={`sponsor-${option.key}`}
            checked={config[option.key] === true}
            onCheckedChange={(enabled) => onChange(option.key, enabled)}
          />
        </div>
      ))}
    </div>
  );
}
