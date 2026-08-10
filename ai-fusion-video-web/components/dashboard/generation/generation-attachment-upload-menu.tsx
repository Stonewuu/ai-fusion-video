"use client";

import { useState } from "react";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import {
  FileVideo2,
  ImageIcon,
  Images,
  Loader2,
  Music2,
  Plus,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type {
  SimpleAttachmentKind,
  SimpleAttachmentUploadOption,
} from "./generation-types";

interface GenerationAttachmentUploadMenuProps {
  options: SimpleAttachmentUploadOption[];
  uploading: boolean;
  disabledReason?: string;
  onSelect: (option: SimpleAttachmentUploadOption) => void;
}

const OPTION_ICONS: Record<SimpleAttachmentKind, typeof ImageIcon> = {
  firstFrame: ImageIcon,
  lastFrame: Images,
  referenceImage: Images,
  referenceVideo: FileVideo2,
  referenceAudio: Music2,
};

export function GenerationAttachmentUploadMenu({
  options,
  uploading,
  disabledReason = "",
  onSelect,
}: GenerationAttachmentUploadMenuProps) {
  const [open, setOpen] = useState(false);
  const selectableOptions = options.filter((option) => !option.disabledReason);
  const disabled = uploading || selectableOptions.length === 0;
  const title =
    disabledReason ||
    options.find((option) => option.disabledReason)?.disabledReason ||
    (options.length === 0 ? "当前模型没有可添加的附件类型" : "添加附件");

  return (
    <PopoverPrimitive.Root open={open} onOpenChange={setOpen}>
      <PopoverPrimitive.Trigger
        render={
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            disabled={disabled}
            title={title}
            aria-label="添加附件"
          />
        }
        openOnHover
        delay={100}
        closeDelay={150}
      >
        {uploading ? (
          <Loader2 className="size-4 animate-spin motion-reduce:animate-none" />
        ) : (
          <Plus className="size-4" />
        )}
      </PopoverPrimitive.Trigger>

      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Positioner
          side="top"
          sideOffset={8}
          align="start"
          className="isolate z-50"
        >
          <PopoverPrimitive.Popup
            initialFocus={false}
            className="w-48 origin-(--transform-origin) rounded-2xl border border-border/40 bg-popover/80 p-2 text-popover-foreground shadow-xl backdrop-blur-xl data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 data-[side=top]:slide-in-from-bottom-2 motion-reduce:transition-none"
          >
            <p className="px-2 pb-1.5 pt-1 text-[10px] font-medium text-muted-foreground">
              添加附件
            </p>
            <div className="grid gap-1" role="menu" aria-label="附件类型">
              {options.map((option) => {
                const Icon = OPTION_ICONS[option.kind];
                return (
                  <Button
                    key={option.kind}
                    type="button"
                    variant="ghost"
                    size="default"
                    role="menuitem"
                    disabled={Boolean(option.disabledReason)}
                    title={option.disabledReason || `上传${option.label}`}
                    onClick={() => {
                      setOpen(false);
                      onSelect(option);
                    }}
                    className="w-full justify-start"
                  >
                    <Icon className="size-4" />
                    <span>{option.label}</span>
                    {option.remaining > 1 && (
                      <span className="ml-auto text-[10px] tabular-nums text-muted-foreground">
                        可传 {option.remaining}
                      </span>
                    )}
                  </Button>
                );
              })}
            </div>
          </PopoverPrimitive.Popup>
        </PopoverPrimitive.Positioner>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
