"use client";

import { useCallback, useEffect, useState } from "react";
import {
  providerCatalogApi,
  type ProviderDefinition,
} from "@/lib/api/provider-catalog";

export function useProviderCatalog() {
  const [providers, setProviders] = useState<ProviderDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setProviders(await providerCatalogApi.list());
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "提供商目录加载失败",
      );
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void reload();
  }, [reload]);
  return { providers, loading, error, reload };
}
