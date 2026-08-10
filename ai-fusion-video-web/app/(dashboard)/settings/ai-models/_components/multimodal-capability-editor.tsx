"use client";

import { FileText, Headphones, ImageIcon, Video } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type {
  MultimodalInputTransport,
  MultimodalInputTransports,
  MultimodalInputType,
} from "@/lib/api/ai-model";

const INPUT_TYPE_OPTIONS = [
  { value: "image", label: "图片", description: "截图、照片与视觉素材", icon: ImageIcon },
  { value: "video", label: "视频", description: "视频文件与动态内容", icon: Video },
  { value: "audio", label: "音频", description: "语音、音乐与声音文件", icon: Headphones },
  { value: "file", label: "文件", description: "PDF 与文本类文档", icon: FileText },
] satisfies Array<{
  value: MultimodalInputType;
  label: string;
  description: string;
  icon: typeof ImageIcon;
}>;

const TRANSPORT_OPTIONS = [
  { value: "url", label: "URL", description: "上传到公开存储后传入地址" },
  { value: "base64", label: "Base64", description: "随请求发送，单文件限 10MB" },
] satisfies Array<{
  value: MultimodalInputTransport;
  label: string;
  description: string;
}>;

interface MultimodalCapabilityEditorProps {
  inputTypes: MultimodalInputType[];
  transports: MultimodalInputTransports;
  onChange: (inputTypes: MultimodalInputType[], transports: MultimodalInputTransports) => void;
}

export function MultimodalCapabilityEditor({
  inputTypes,
  transports,
  onChange,
}: MultimodalCapabilityEditorProps) {
  const toggleType = (type: MultimodalInputType) => {
    if (inputTypes.includes(type)) {
      const nextTransports = { ...transports };
      delete nextTransports[type];
      onChange(inputTypes.filter((item) => item !== type), nextTransports);
      return;
    }
    onChange([...inputTypes, type], { ...transports, [type]: ["base64"] });
  };

  const toggleTransport = (type: MultimodalInputType, transport: MultimodalInputTransport) => {
    const current = transports[type] ?? [];
    const next = current.includes(transport)
      ? current.filter((item) => item !== transport)
      : [...current, transport];
    if (!next.length) return;
    onChange(inputTypes, { ...transports, [type]: next });
  };

  return (
    <div className="space-y-3">
      <div>
        <Label className="text-xs text-muted-foreground">多模态输入</Label>
        <p className="mt-1 text-[10px] leading-4 text-muted-foreground">
          只声明模型与当前接入协议实际支持的类型；助手会据此决定上传方式。
        </p>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {INPUT_TYPE_OPTIONS.map(({ value, label, description, icon: Icon }) => {
          const checked = inputTypes.includes(value);
          return (
            <label
              key={value}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-lg border border-border/30 bg-background/50 p-3 transition-colors",
                "hover:border-border/50",
                checked && "border-primary/40 bg-primary/5",
              )}
            >
              <Checkbox checked={checked} onCheckedChange={() => toggleType(value)} />
              <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
              <span className="min-w-0">
                <span className="block text-xs font-medium">{label}</span>
                <span className="mt-0.5 block text-[10px] leading-4 text-muted-foreground">{description}</span>
              </span>
            </label>
          );
        })}
      </div>

      {inputTypes.length ? (
        <div className="divide-y divide-border/20 border-y border-border/20">
          {inputTypes.map((type) => {
            const option = INPUT_TYPE_OPTIONS.find((item) => item.value === type)!;
            return (
              <div key={type} className="flex flex-wrap items-center gap-3 py-2.5">
                <span className="w-14 shrink-0 text-xs font-medium">{option.label}</span>
                <div className="flex flex-wrap items-center gap-4">
                  {TRANSPORT_OPTIONS.map((transport) => (
                    <label key={transport.value} className="flex cursor-pointer items-center gap-2">
                      <Checkbox
                        checked={(transports[type] ?? []).includes(transport.value)}
                        onCheckedChange={() => toggleTransport(type, transport.value)}
                      />
                      <span>
                        <span className="block text-[11px] font-medium">{transport.label}</span>
                        <span className="block text-[9px] text-muted-foreground">{transport.description}</span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-[10px] text-muted-foreground">当前模型仅接受文本输入。</p>
      )}
    </div>
  );
}
