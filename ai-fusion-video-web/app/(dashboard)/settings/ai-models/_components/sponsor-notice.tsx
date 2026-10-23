import { ExternalLink } from "lucide-react";
import type { ProviderDefinition } from "@/lib/api/provider-catalog";

export function SponsorNotice({ provider }: { provider: ProviderDefinition }) {
  if (!provider.sponsored) {
    return null;
  }
  return (
    <p className="flex flex-wrap items-center gap-1 text-xs leading-5 text-muted-foreground">
      点击
      <a
        href={provider.credentialUrl}
        target="_blank"
        rel="sponsored noopener noreferrer"
        className="inline-flex items-center gap-1 rounded-sm text-primary underline underline-offset-4 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"
      >
        {provider.label}
        <ExternalLink className="size-3" aria-hidden="true" />
      </a>
      前往平台获取密钥
    </p>
  );
}
