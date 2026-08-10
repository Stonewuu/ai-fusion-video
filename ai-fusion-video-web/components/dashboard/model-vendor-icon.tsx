"use client";

import AgnesAI from "@lobehub/icons/es/AgnesAI/components/Mono";
import Anthropic from "@lobehub/icons/es/Anthropic/components/Mono";
import Bailian from "@lobehub/icons/es/Bailian/components/Color";
import Claude from "@lobehub/icons/es/Claude/components/Color";
import DeepMind from "@lobehub/icons/es/DeepMind/components/Color";
import DeepSeek from "@lobehub/icons/es/DeepSeek/components/Color";
import Doubao from "@lobehub/icons/es/Doubao/components/Color";
import Gemini from "@lobehub/icons/es/Gemini/components/Color";
import Google from "@lobehub/icons/es/Google/components/Color";
import Jimeng from "@lobehub/icons/es/Jimeng/components/Color";
import Kling from "@lobehub/icons/es/Kling/components/Color";
import Moonshot from "@lobehub/icons/es/Moonshot/components/Mono";
import NanoBanana from "@lobehub/icons/es/NanoBanana/components/Color";
import NewAPI from "@lobehub/icons/es/NewAPI/components/Color";
import Ollama from "@lobehub/icons/es/Ollama/components/Mono";
import OpenAI from "@lobehub/icons/es/OpenAI/components/Mono";
import Qwen from "@lobehub/icons/es/Qwen/components/Color";
import SiliconCloud from "@lobehub/icons/es/SiliconCloud/components/Color";
import Sora from "@lobehub/icons/es/Sora/components/Color";
import VertexAI from "@lobehub/icons/es/VertexAI/components/Color";
import Volcengine from "@lobehub/icons/es/Volcengine/components/Color";
import Zhipu from "@lobehub/icons/es/Zhipu/components/Color";
import type { IconType } from "@lobehub/icons/es/types";
import { Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ModelIconSource {
  name?: string | null;
  code?: string | null;
  platform?: string | null;
  capabilityPresetCode?: string | null;
}

interface LobeIconDefinition {
  Icon: IconType;
}

const LOBE_ICONS = {
  agnes: { Icon: AgnesAI },
  anthropic: { Icon: Anthropic },
  bailian: { Icon: Bailian },
  claude: { Icon: Claude },
  deepMind: { Icon: DeepMind },
  deepSeek: { Icon: DeepSeek },
  doubao: { Icon: Doubao },
  gemini: { Icon: Gemini },
  google: { Icon: Google },
  jimeng: { Icon: Jimeng },
  kling: { Icon: Kling },
  moonshot: { Icon: Moonshot },
  nanoBanana: { Icon: NanoBanana },
  newApi: { Icon: NewAPI },
  ollama: { Icon: Ollama },
  openAi: { Icon: OpenAI },
  qwen: { Icon: Qwen },
  siliconCloud: { Icon: SiliconCloud },
  sora: { Icon: Sora },
  vertexAi: { Icon: VertexAI },
  volcengine: { Icon: Volcengine },
  zhipu: { Icon: Zhipu },
} satisfies Record<string, LobeIconDefinition>;

function normalizedIdentity(values: Array<string | null | undefined>) {
  return values
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join(" ")
    .toLocaleLowerCase();
}

function resolveModelIcon(source?: ModelIconSource | null): LobeIconDefinition | null {
  if (!source) return null;
  const identity = normalizedIdentity([
    source.capabilityPresetCode,
    source.code,
    source.name,
  ]);

  if (identity.includes("agnes")) return LOBE_ICONS.agnes;
  if (identity.includes("sora")) return LOBE_ICONS.sora;
  if (identity.includes("nano-banana")
    || /gemini-[^\s]*(?:flash|pro)-image/.test(identity)) {
    return LOBE_ICONS.nanoBanana;
  }
  if (identity.includes("claude")) return LOBE_ICONS.claude;
  if (identity.includes("anthropic")) return LOBE_ICONS.anthropic;
  if (identity.includes("deepseek")) return LOBE_ICONS.deepSeek;
  if (identity.includes("kling")) return LOBE_ICONS.kling;
  if (identity.includes("jimeng")
    || identity.includes("seedream")
    || identity.includes("seedance")) {
    return LOBE_ICONS.jimeng;
  }
  if (identity.includes("doubao")) return LOBE_ICONS.doubao;
  if (identity.includes("veo_") || identity.includes("veo-")) return LOBE_ICONS.vertexAi;
  if (identity.includes("imagen")) return LOBE_ICONS.deepMind;
  if (identity.includes("gemini")) return LOBE_ICONS.gemini;
  if (identity.includes("qwen")
    || identity.includes("wanx")
    || /(?:^|[\s/_-])wan\d/.test(identity)) {
    return LOBE_ICONS.qwen;
  }
  if (identity.includes("gpt-") || identity.includes("openai")) return LOBE_ICONS.openAi;

  return resolveProviderIcon(source.platform);
}

function resolveProviderIcon(provider?: string | null): LobeIconDefinition | null {
  const identity = normalizedIdentity([provider]);
  if (!identity) return null;

  if (identity.includes("agnes")) return LOBE_ICONS.agnes;
  if (identity.includes("deepseek")) return LOBE_ICONS.openAi;
  if (identity.includes("newapi") || identity.includes("new api")) return LOBE_ICONS.newApi;
  if (identity.includes("zhipu") || identity.includes("智谱")) return LOBE_ICONS.zhipu;
  if (identity.includes("moonshot") || identity.includes("月之暗面")) return LOBE_ICONS.moonshot;
  if (identity.includes("siliconflow") || identity.includes("silicon flow")
    || identity.includes("硅基流动")) {
    return LOBE_ICONS.siliconCloud;
  }
  if (identity.includes("vertex")) return LOBE_ICONS.vertexAi;
  if (identity.includes("googleflow") || identity.includes("google flow")) return LOBE_ICONS.google;
  if (identity.includes("gemini")) return LOBE_ICONS.gemini;
  if (identity.includes("dashscope") || identity.includes("bailian") || identity.includes("百炼")) {
    return LOBE_ICONS.bailian;
  }
  if (identity.includes("anthropic") || identity.includes("claude")) return LOBE_ICONS.anthropic;
  if (identity.includes("ollama")) return LOBE_ICONS.ollama;
  if (identity.includes("volcengine") || identity.includes("火山")) return LOBE_ICONS.volcengine;
  if (identity.includes("doubao") || identity.includes("豆包")) return LOBE_ICONS.doubao;
  if (identity.includes("openai") || identity.includes("openai_compatible")) return LOBE_ICONS.openAi;
  return null;
}

function LobeIcon({
  definition,
  className,
}: {
  definition: LobeIconDefinition | null;
  className?: string;
}) {
  if (!definition) {
    return <Cpu className={cn("size-4 shrink-0 text-muted-foreground", className)} />;
  }

  const { Icon } = definition;
  return (
    <Icon
      aria-hidden="true"
      className={cn("size-4 shrink-0", className)}
      size="1em"
    />
  );
}

export function ModelVendorIcon({
  source,
  className,
}: {
  source?: ModelIconSource | null;
  className?: string;
}) {
  return <LobeIcon definition={resolveModelIcon(source)} className={className} />;
}

export function ProviderVendorIcon({
  provider,
  className,
}: {
  provider?: string | null;
  className?: string;
}) {
  return (
    <LobeIcon
      definition={resolveProviderIcon(provider)}
      className={className}
    />
  );
}
