"use client";

import { useState } from "react";
import type { ModelPreset } from "@/lib/api/ai-model";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CUSTOM_CAPABILITY_PRESET_VALUE } from "./model-config-support";

export function CapabilityPresetSelect({
  selectedCode,
  presets,
  hasCustomConfig,
  onSelectCustom,
  onApplyPreset,
}: {
  selectedCode?: string;
  presets: ModelPreset[];
  hasCustomConfig: boolean;
  onSelectCustom: () => void;
  onApplyPreset: (preset: ModelPreset) => void;
}) {
  const [pendingPreset, setPendingPreset] = useState<ModelPreset | null>(null);
  const options = [
    { value: CUSTOM_CAPABILITY_PRESET_VALUE, label: "自定义" },
    ...presets.map(preset => ({ value: preset.code, label: preset.name })),
  ];

  const handleValueChange = (rawValue: unknown) => {
    const value = String(rawValue);
    if (value === CUSTOM_CAPABILITY_PRESET_VALUE) {
      onSelectCustom();
      return;
    }

    const preset = presets.find(item => item.code === value);
    if (!preset || preset.code === selectedCode) return;
    if (!selectedCode && hasCustomConfig) {
      setPendingPreset(preset);
      return;
    }
    onApplyPreset(preset);
  };

  const confirmApply = () => {
    if (!pendingPreset) return;
    onApplyPreset(pendingPreset);
    setPendingPreset(null);
  };

  return (
    <>
      <div className="space-y-1.5">
        <Label className="text-xs text-muted-foreground">模型能力预设</Label>
        <Select
          value={selectedCode || CUSTOM_CAPABILITY_PRESET_VALUE}
          onValueChange={handleValueChange}
          items={options}
        >
          <SelectTrigger className="w-full text-sm">
            <SelectValue placeholder="选择能力预设" />
          </SelectTrigger>
          <SelectContent className="text-sm">
            <SelectGroup>
              <SelectItem value={CUSTOM_CAPABILITY_PRESET_VALUE} className="text-sm">
                自定义
              </SelectItem>
              {presets.map(preset => (
                <SelectItem key={preset.code} value={preset.code} className="text-sm">
                  {preset.name}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
        <p className="text-[10px] leading-4 text-muted-foreground">
          预设会载入高级能力字段；手动修改后自动切换为自定义。请求协议独立配置，不由能力预设推断。
        </p>
      </div>

      <Dialog
        open={pendingPreset !== null}
        onOpenChange={nextOpen => {
          if (!nextOpen) setPendingPreset(null);
        }}
      >
        <DialogPortal>
          <DialogOverlay
            forceRender
            className="z-[60] bg-black/55 supports-backdrop-filter:backdrop-blur-sm"
          />
        </DialogPortal>
        <DialogContent className="z-[70] rounded-2xl border border-border/40 bg-popover/95 shadow-2xl ring-0 backdrop-blur-xl sm:max-w-md">
          <DialogHeader>
            <DialogTitle>覆盖自定义能力设置？</DialogTitle>
            <DialogDescription>
              切换到「{pendingPreset?.name}」会清除当前自定义高级能力配置，并载入该预设。模型标识和请求协议不会改变。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" size="sm" />}>取消</DialogClose>
            <Button size="sm" onClick={confirmApply}>覆盖并切换</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
