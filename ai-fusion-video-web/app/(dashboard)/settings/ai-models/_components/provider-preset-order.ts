import type { ProviderDefinition } from "@/lib/api/provider-catalog";

const SPONSOR_ORDER = ["infistar", "metaso", "erchun"];

export function orderProviderPresets(providers: readonly ProviderDefinition[]) {
  const sponsors = SPONSOR_ORDER.flatMap((id) =>
    providers.filter((provider) => provider.sponsored && provider.id === id),
  );
  const orderedIds = new Set(sponsors.map((provider) => provider.id));
  return [
    ...sponsors,
    ...providers.filter((provider) => !orderedIds.has(provider.id)),
  ];
}
