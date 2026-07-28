"use client";

import { Cpu } from "lucide-react";
import { SafeImage } from "@/components/ui/safe-image";
import { cn } from "@/lib/utils";

export interface ModelIconSource {
  name?: string | null;
  code?: string | null;
  icon?: string | null;
  platform?: string | null;
  capabilityPresetCode?: string | null;
}

interface VendorIconDefinition {
  label: string;
  src: string;
  monochrome?: boolean;
  preferCurated?: boolean;
}

const VENDOR_ICONS = {
  agnes: { label: "Agnes AI", src: "/model-vendors/agnes.svg" },
  anthropic: { label: "Anthropic Claude", src: "/model-vendors/claude.svg" },
  deepseek: { label: "DeepSeek", src: "/model-vendors/deepseek.svg" },
  doubao: { label: "豆包", src: "/model-vendors/doubao.svg" },
  gemini: { label: "Google Gemini", src: "/model-vendors/gemini.svg", preferCurated: true },
  jimeng: { label: "即梦", src: "/model-vendors/jimeng.svg" },
  kling: { label: "可灵", src: "/model-vendors/kling.svg" },
  openai: { label: "OpenAI", src: "/model-vendors/openai.svg", monochrome: true },
  qwen: { label: "阿里通义", src: "/model-vendors/qwen.svg" },
  volcengine: { label: "火山引擎", src: "/model-vendors/volcengine.svg" },
} satisfies Record<string, VendorIconDefinition>;

function resolveVendor(source?: ModelIconSource | null): VendorIconDefinition | null {
  if (!source) return null;
  const identity = [
    source.capabilityPresetCode,
    source.code,
    source.name,
    source.platform,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (identity.includes("agnes")) return VENDOR_ICONS.agnes;
  if (identity.includes("anthropic") || identity.includes("claude")) return VENDOR_ICONS.anthropic;
  if (identity.includes("deepseek")) return VENDOR_ICONS.deepseek;
  if (identity.includes("kling") || identity.includes("可灵")) return VENDOR_ICONS.kling;
  if (identity.includes("jimeng") || identity.includes("即梦")) return VENDOR_ICONS.jimeng;
  if (identity.includes("doubao") || identity.includes("seedream") || identity.includes("seedance")) {
    return VENDOR_ICONS.doubao;
  }
  if (identity.includes("volcengine") || identity.includes("火山")) return VENDOR_ICONS.volcengine;
  if (
    identity.includes("gemini")
    || identity.includes("imagen")
    || identity.includes("veo_")
    || identity.includes("googleflow")
    || identity.includes("google flow")
    || identity.includes("vertex_ai")
    || identity.includes("vertexai")
  ) {
    return VENDOR_ICONS.gemini;
  }
  if (
    identity.includes("qwen")
    || identity.includes("wan2")
    || identity.includes("wanx")
    || identity.includes("dashscope")
  ) {
    return VENDOR_ICONS.qwen;
  }
  if (
    identity.includes("openai")
    || identity.includes("gpt-image")
    || identity.includes("sora")
  ) {
    return VENDOR_ICONS.openai;
  }
  return null;
}

export function ModelVendorIcon({
  source,
  className,
}: {
  source?: ModelIconSource | null;
  className?: string;
}) {
  const vendor = resolveVendor(source);
  const src = vendor?.preferCurated ? vendor.src : source?.icon?.trim() || vendor?.src;
  const fallback = <Cpu className={cn("size-4 text-muted-foreground", className)} />;

  if (!src) return fallback;

  return (
    <SafeImage
      src={src}
      alt=""
      title={vendor?.label || source?.name || "模型"}
      className={cn(
        "size-4 shrink-0 object-contain",
        vendor?.monochrome && (!source?.icon || vendor.preferCurated) && "dark:invert",
        className,
      )}
      fallback={fallback}
    />
  );
}
