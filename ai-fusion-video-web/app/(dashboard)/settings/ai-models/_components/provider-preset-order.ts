import type { ProviderDefinition } from "@/lib/api/provider-catalog";

export const DEFAULT_PROVIDER_PRESET_LIMIT = 8;

const PROVIDER_ORDER = [
  "infistar",
  "metaso",
  "erchun",
  "deepseek",
  "openai",
  "comfyui",
  "dashscope",
  "minimax-cn",
  "minimax-global",
  "volcengine",
];

const providerRanks = new Map(PROVIDER_ORDER.map((id, index) => [id, index]));

export function orderProviderPresets(providers: readonly ProviderDefinition[]) {
  return [...providers].sort((left, right) => {
    const sponsorship = Number(right.sponsored) - Number(left.sponsored);
    if (sponsorship !== 0) {
      return sponsorship;
    }
    const leftRank = providerRanks.get(left.id) ?? PROVIDER_ORDER.length;
    const rightRank = providerRanks.get(right.id) ?? PROVIDER_ORDER.length;
    return leftRank - rightRank;
  });
}
