import {
  MODEL_TYPE_OPTIONS,
  PLATFORM_LABELS,
  type AiModel,
  type ApiConfig,
  type RemoteModel,
} from "@/lib/api/ai-model";

export type AiModelStatusFilter = "all" | "enabled" | "disabled";

export interface FilteredConfigEntry {
  config: ApiConfig;
  allModels: AiModel[];
  visibleModels: AiModel[];
  hiddenModelCount: number;
  matchedByConfig: boolean;
}

export interface DefaultModelSlot {
  modelType: number;
  label: string;
  model: AiModel | null;
  config: ApiConfig | null;
  unavailableReason: string | null;
  candidates: Array<{ model: AiModel; config: ApiConfig }>;
}

export function filterApiConfigsAndModels(
  configs: readonly ApiConfig[],
  models: readonly AiModel[],
  options: {
    query: string;
    modelType: number;
    status: AiModelStatusFilter;
  },
): FilteredConfigEntry[] {
  const key = options.query.trim().toLowerCase();
  const hasTypeFilter = options.modelType !== 0;

  const entries: FilteredConfigEntry[] = [];
  for (const config of configs) {
    const configModels = models.filter((model) => model.apiConfigId === config.id);
    const configSearchText = `${config.name} ${config.platform ?? ""} ${PLATFORM_LABELS[config.platform ?? ""] ?? ""}`.toLowerCase();
    const matchedByConfig = key.length > 0 && configSearchText.includes(key);

    const statusMatchedConfig =
      options.status === "all"
      || (options.status === "enabled" && config.status === 1)
      || (options.status === "disabled" && config.status === 0);

    const matchesModelStatus = (model: AiModel) => {
      if (options.status === "all") {
        return true;
      }
      const effectiveEnabled = config.status === 1 && model.status === 1;
      if (options.status === "enabled") {
        return effectiveEnabled;
      }
      return !effectiveEnabled;
    };

    const matchesModelType = (model: AiModel) => {
      if (!hasTypeFilter) {
        return true;
      }
      return model.modelType === options.modelType;
    };

    const matchesModelQuery = (model: AiModel) => {
      if (!key || matchedByConfig) {
        return true;
      }
      return `${model.name} ${model.code}`.toLowerCase().includes(key);
    };

    const visibleModels = configModels.filter((model) =>
      matchesModelType(model) && matchesModelStatus(model) && matchesModelQuery(model),
    );

    const hasAnyFilter = key.length > 0 || hasTypeFilter || options.status !== "all";
    if (!hasAnyFilter) {
      entries.push({
        config,
        allModels: configModels,
        visibleModels: configModels,
        hiddenModelCount: 0,
        matchedByConfig: false,
      });
      continue;
    }

    if (hasTypeFilter) {
      if (visibleModels.length === 0) {
        continue;
      }
    } else if (key.length > 0) {
      if (!matchedByConfig && visibleModels.length === 0) {
        continue;
      }
      if (matchedByConfig && !statusMatchedConfig && visibleModels.length === 0) {
        continue;
      }
    } else if (!statusMatchedConfig && visibleModels.length === 0) {
      continue;
    }

    entries.push({
      config,
      allModels: configModels,
      visibleModels,
      hiddenModelCount: Math.max(0, configModels.length - visibleModels.length),
      matchedByConfig,
    });
  }

  return entries;
}

export function resolveDefaultModelsByType(
  models: readonly AiModel[],
  configs: readonly ApiConfig[],
): DefaultModelSlot[] {
  const configMap = new Map(configs.map((config) => [config.id, config]));

  return MODEL_TYPE_OPTIONS.map((option) => {
    const typeModels = models.filter((model) => model.modelType === option.value);
    const defaultModel = typeModels.find((model) => model.defaultModel) ?? null;
    const defaultConfig = defaultModel?.apiConfigId
      ? configMap.get(defaultModel.apiConfigId) ?? null
      : null;

    let unavailableReason: string | null = null;
    if (defaultModel) {
      if (defaultModel.status === 0) {
        unavailableReason = "模型已停用";
      } else if (!defaultConfig) {
        unavailableReason = "关联服务不存在";
      } else if (defaultConfig.status === 0) {
        unavailableReason = "所属服务已停用";
      }
    }

    const candidates: Array<{ model: AiModel; config: ApiConfig }> = [];
    for (const model of typeModels) {
      if (model.status !== 1 || !model.apiConfigId) {
        continue;
      }
      const config = configMap.get(model.apiConfigId);
      if (!config || config.status !== 1) {
        continue;
      }
      candidates.push({ model, config });
    }

    return {
      modelType: option.value,
      label: option.label,
      model: defaultModel,
      config: defaultConfig,
      unavailableReason,
      candidates,
    };
  });
}

export function filterRemoteModels(
  remoteModels: readonly RemoteModel[],
  existingModelCodes: ReadonlySet<string>,
  options: {
    query: string;
    modelType: number;
    hideExisting: boolean;
  },
): RemoteModel[] {
  const key = options.query.trim().toLowerCase();
  return remoteModels.filter((model) => {
    if (options.hideExisting && existingModelCodes.has(model.id)) {
      return false;
    }
    if (options.modelType !== 0 && model.modelType !== options.modelType) {
      return false;
    }
    if (!key) {
      return true;
    }
    const haystack = `${model.id} ${model.displayName ?? ""} ${model.ownedBy ?? ""}`.toLowerCase();
    return haystack.includes(key);
  });
}
