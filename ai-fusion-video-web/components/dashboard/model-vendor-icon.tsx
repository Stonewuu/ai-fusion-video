"use client";

import Image from "next/image";
import AgnesAI from "@lobehub/icons/es/AgnesAI/components/Mono";
import Ai360 from "@lobehub/icons/es/Ai360/components/Mono";
import Anthropic from "@lobehub/icons/es/Anthropic/components/Mono";
import Baichuan from "@lobehub/icons/es/Baichuan/components/Color";
import Bailian from "@lobehub/icons/es/Bailian/components/Color";
import Claude from "@lobehub/icons/es/Claude/components/Color";
import Cohere from "@lobehub/icons/es/Cohere/components/Color";
import ComfyUI from "@lobehub/icons/es/ComfyUI/components/Color";
import DeepMind from "@lobehub/icons/es/DeepMind/components/Color";
import DeepSeek from "@lobehub/icons/es/DeepSeek/components/Color";
import Doubao from "@lobehub/icons/es/Doubao/components/Color";
import Flux from "@lobehub/icons/es/Flux/components/Mono";
import Gemini from "@lobehub/icons/es/Gemini/components/Color";
import Google from "@lobehub/icons/es/Google/components/Color";
import Grok from "@lobehub/icons/es/Grok/components/Mono";
import Hailuo from "@lobehub/icons/es/Hailuo/components/Color";
import Hunyuan from "@lobehub/icons/es/Hunyuan/components/Color";
import Jimeng from "@lobehub/icons/es/Jimeng/components/Color";
import Kimi from "@lobehub/icons/es/Kimi/components/Color";
import Kling from "@lobehub/icons/es/Kling/components/Color";
import Luma from "@lobehub/icons/es/Luma/components/Color";
import Meta from "@lobehub/icons/es/Meta/components/Color";
import Minimax from "@lobehub/icons/es/Minimax/components/Color";
import Mistral from "@lobehub/icons/es/Mistral/components/Color";
import Moonshot from "@lobehub/icons/es/Moonshot/components/Mono";
import NanoBanana from "@lobehub/icons/es/NanoBanana/components/Color";
import NewAPI from "@lobehub/icons/es/NewAPI/components/Color";
import Ollama from "@lobehub/icons/es/Ollama/components/Mono";
import OpenAI from "@lobehub/icons/es/OpenAI/components/Mono";
import PixVerse from "@lobehub/icons/es/PixVerse/components/Color";
import Qwen from "@lobehub/icons/es/Qwen/components/Color";
import Runway from "@lobehub/icons/es/Runway/components/Mono";
import SiliconCloud from "@lobehub/icons/es/SiliconCloud/components/Color";
import Sora from "@lobehub/icons/es/Sora/components/Color";
import Stability from "@lobehub/icons/es/Stability/components/Color";
import Stepfun from "@lobehub/icons/es/Stepfun/components/Mono";
import VertexAI from "@lobehub/icons/es/VertexAI/components/Color";
import Vidu from "@lobehub/icons/es/Vidu/components/Color";
import Volcengine from "@lobehub/icons/es/Volcengine/components/Color";
import Wenxin from "@lobehub/icons/es/Wenxin/components/Color";
import XiaomiMiMo from "@lobehub/icons/es/XiaomiMiMo/components/Mono";
import Yi from "@lobehub/icons/es/Yi/components/Color";
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

type BrandIconDefinition = { Icon: IconType } | { src: string };

const BRAND_ICONS = {
  agnes: { Icon: AgnesAI },
  ai360: { Icon: Ai360 },
  anthropic: { Icon: Anthropic },
  baichuan: { Icon: Baichuan },
  bailian: { Icon: Bailian },
  claude: { Icon: Claude },
  cohere: { Icon: Cohere },
  comfyui: { Icon: ComfyUI },
  deepMind: { Icon: DeepMind },
  deepSeek: { Icon: DeepSeek },
  doubao: { Icon: Doubao },
  flux: { Icon: Flux },
  gemini: { Icon: Gemini },
  google: { Icon: Google },
  grok: { Icon: Grok },
  hailuo: { Icon: Hailuo },
  hunyuan: { Icon: Hunyuan },
  infistar: { src: "/brands/infistar.png" },
  jimeng: { Icon: Jimeng },
  kimi: { Icon: Kimi },
  kling: { Icon: Kling },
  luma: { Icon: Luma },
  meta: { Icon: Meta },
  metaso: { src: "/brands/metaso.png" },
  minimax: { Icon: Minimax },
  mistral: { Icon: Mistral },
  moonshot: { Icon: Moonshot },
  nanoBanana: { Icon: NanoBanana },
  newApi: { Icon: NewAPI },
  ollama: { Icon: Ollama },
  openAi: { Icon: OpenAI },
  pixverse: { Icon: PixVerse },
  qwen: { Icon: Qwen },
  runway: { Icon: Runway },
  siliconCloud: { Icon: SiliconCloud },
  sora: { Icon: Sora },
  stability: { Icon: Stability },
  stepfun: { Icon: Stepfun },
  vertexAi: { Icon: VertexAI },
  vidu: { Icon: Vidu },
  volcengine: { Icon: Volcengine },
  wenxin: { Icon: Wenxin },
  mimo: { Icon: XiaomiMiMo },
  yi: { Icon: Yi },
  zhipu: { Icon: Zhipu },
} satisfies Record<string, BrandIconDefinition>;

type IconRule = [RegExp, keyof typeof BRAND_ICONS];

// 仅用于品牌展示；请求协议和模型能力始终由显式配置决定。
const MODEL_ICON_RULES: IconRule[] = [
  [/agnes/, "agnes"],
  [/sora/, "sora"],
  [/nano[- ]?banana|gemini-[^\s]*(?:flash|pro)(?:-lite)?-image/, "nanoBanana"],
  [/claude/, "claude"],
  [/anthropic/, "anthropic"],
  [/deepseek|深度求索/, "deepSeek"],
  [/minimax|mini-max/, "minimax"],
  [/hailuo|海螺/, "hailuo"],
  [/kling|可灵/, "kling"],
  [/jimeng|seedream|seedance|即梦/, "jimeng"],
  [/doubao|豆包/, "doubao"],
  [/veo[_-]/, "vertexAi"],
  [/imagen/, "deepMind"],
  [/gemini/, "gemini"],
  [/qwen|wanx|(?:^|[\s/:_-])wan\d|千问|万相/, "qwen"],
  [/gpt-|openai|(?:^|[\s/:])o[134](?:[-\s]|$)/, "openAi"],
  [/kimi/, "kimi"],
  [/moonshot|月之暗面/, "moonshot"],
  [/grok/, "grok"],
  [/glm|zhipu|智谱/, "zhipu"],
  [/hunyuan|(?:^|[\s/:])hy\d|混元/, "hunyuan"],
  [/stepfun|(?:^|[\s/:])step-|阶跃/, "stepfun"],
  [/mimo/, "mimo"],
  [/baichuan|百川/, "baichuan"],
  [/ernie|wenxin|文心/, "wenxin"],
  [/360zhinao|智脑/, "ai360"],
  [/mistral|mixtral|codestral|devstral|pixtral|ministral/, "mistral"],
  [/llama/, "meta"],
  [/flux/, "flux"],
  [/stable[-_ ]diffusion|sdxl|(?:^|[\s/:])sd[23]/, "stability"],
  [/vidu/, "vidu"],
  [/pixverse/, "pixverse"],
  [/luma|dream-machine|(?:^|[\s/:])ray[23-]/, "luma"],
  [/runway|gen-[34]/, "runway"],
  [/cohere|command-r/, "cohere"],
  [/(?:^|[\s/:])yi-/, "yi"],
];

const PROVIDER_ICON_RULES: IconRule[] = [
  [/metaso|秘塔/, "metaso"],
  [/infistar|无限星河/, "infistar"],
  [/minimax/, "minimax"],
  [/comfyui/, "comfyui"],
  [/agnes/, "agnes"],
  [/deepseek/, "deepSeek"],
  [/newapi|new api/, "newApi"],
  [/zhipu|智谱/, "zhipu"],
  [/moonshot|月之暗面/, "moonshot"],
  [/siliconflow|silicon flow|硅基流动/, "siliconCloud"],
  [/vertex/, "vertexAi"],
  [/googleflow|google flow/, "google"],
  [/gemini/, "gemini"],
  [/dashscope|bailian|百炼|wan3/, "bailian"],
  [/anthropic|claude/, "anthropic"],
  [/ollama/, "ollama"],
  [/volcengine|火山/, "volcengine"],
  [/doubao|豆包/, "doubao"],
  [/openai/, "openAi"],
];

function normalizedIdentity(values: Array<string | null | undefined>) {
  return values
    .filter(
      (value): value is string =>
        typeof value === "string" && value.trim().length > 0,
    )
    .join(" ")
    .toLocaleLowerCase();
}

function matchIcon(
  identity: string,
  rules: IconRule[],
): BrandIconDefinition | null {
  const rule = rules.find(([pattern]) => pattern.test(identity));
  return rule ? BRAND_ICONS[rule[1]] : null;
}

function resolveProviderIcon(provider?: string | null) {
  return matchIcon(normalizedIdentity([provider]), PROVIDER_ICON_RULES);
}

function resolveModelIcon(
  source?: ModelIconSource | null,
): BrandIconDefinition | null {
  if (!source) {
    return null;
  }
  const identity = normalizedIdentity([
    source.capabilityPresetCode,
    source.code,
    source.name,
  ]);
  return (
    matchIcon(identity, MODEL_ICON_RULES) ??
    resolveProviderIcon(source.platform)
  );
}

function BrandIcon({
  definition,
  className,
}: {
  definition: BrandIconDefinition | null;
  className?: string;
}) {
  if (!definition) {
    return (
      <Cpu
        aria-hidden="true"
        className={cn("size-4 shrink-0 text-muted-foreground", className)}
      />
    );
  }
  if ("src" in definition) {
    return (
      <Image
        src={definition.src}
        alt=""
        aria-hidden="true"
        width={24}
        height={24}
        sizes="32px"
        className={cn("size-4 shrink-0 object-contain", className)}
      />
    );
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
  return (
    <BrandIcon definition={resolveModelIcon(source)} className={className} />
  );
}

export function ProviderVendorIcon({
  provider,
  className,
}: {
  provider?: string | null;
  className?: string;
}) {
  return (
    <BrandIcon
      definition={resolveProviderIcon(provider)}
      className={className}
    />
  );
}
