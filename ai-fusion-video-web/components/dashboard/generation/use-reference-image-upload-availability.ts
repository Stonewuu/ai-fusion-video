"use client";

import { useEffect, useMemo, useState } from "react";
import { http } from "@/lib/api/client";
import { storageConfigApi } from "@/lib/api/storage";
import { toastApiError } from "@/lib/api/toast-api-error";
import {
  resolveReferenceImageUploadAvailability,
  type GenerationCapabilities,
} from "@/lib/generation-capabilities";
import type { WorkbenchMode } from "./generation-types";

export function useReferenceImageUploadAvailability(
  mode: WorkbenchMode,
  capabilities: GenerationCapabilities | null,
) {
  const [runtimeConfig, setRuntimeConfig] = useState({
    loaded: false,
    publicMediaAccessConfigured: false,
  });

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      http.get<never, string | null>("/api/system/config/resource_base_url"),
      storageConfigApi.list(),
    ])
      .then(([resourceBaseUrl, storageConfigs]) => {
        if (cancelled) return;
        const defaultStorage = storageConfigs.find(
          (config) => config.isDefault && config.status === 1,
        );
        const publicObjectStorage = Boolean(
          defaultStorage &&
            defaultStorage.type === "s3",
        );
        setRuntimeConfig({
          loaded: true,
          publicMediaAccessConfigured:
            Boolean(resourceBaseUrl?.trim()) || publicObjectStorage,
        });
      })
      .catch((error) => {
        if (!cancelled) {
          toastApiError(error, "加载参考图上传配置失败");
          setRuntimeConfig({
            loaded: true,
            publicMediaAccessConfigured: false,
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [mode]);

  return useMemo(
    () =>
      resolveReferenceImageUploadAvailability(
        capabilities,
        runtimeConfig.publicMediaAccessConfigured,
        runtimeConfig.loaded,
      ),
    [capabilities, runtimeConfig],
  );
}
