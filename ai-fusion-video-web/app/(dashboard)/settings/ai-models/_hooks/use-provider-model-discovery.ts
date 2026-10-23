"use client";

import { useCallback, useRef, useState } from "react";
import type { ApiConfigSaveReq } from "@/lib/api/ai-model";
import {
  providerCatalogApi,
  type ProviderDiscoveryResult,
} from "@/lib/api/provider-catalog";

export function useProviderModelDiscovery() {
  const [result, setResult] = useState<ProviderDiscoveryResult | null>(null);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<Promise<boolean> | null>(null);

  const read = useCallback(
    (configuration: ApiConfigSaveReq): Promise<boolean> => {
      // 自动读取与开发模式重复挂载共用在途请求，避免同一弹窗重复请求上游。
      if (pending.current) {
        return pending.current;
      }
      setReading(true);
      setError(null);
      pending.current = (async () => {
        try {
          setResult(await providerCatalogApi.preview(configuration));
          return true;
        } catch (failure) {
          setError(failure instanceof Error ? failure.message : "读取模型失败");
          return false;
        } finally {
          pending.current = null;
          setReading(false);
        }
      })();
      return pending.current;
    },
    [],
  );

  const clear = () => {
    setResult(null);
    setError(null);
  };

  return { result, reading, error, read, clear };
}
