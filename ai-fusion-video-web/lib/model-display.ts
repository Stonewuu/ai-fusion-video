export interface ModelDisplaySource {
  name?: string | null;
  code?: string | null;
}

export interface ModelDisplayParts {
  name: string;
  code?: string;
}

function normalizedText(value: string | null | undefined) {
  const normalized = value?.trim();
  return normalized || undefined;
}

/**
 * 统一模型标识展示：主信息优先使用模型名称，名称缺失时回退到编码；
 * 编码仅在与主信息不同时作为辅助信息展示。
 */
export function getModelDisplayParts(
  model: ModelDisplaySource | null | undefined,
  fallback = "未命名模型",
): ModelDisplayParts {
  const rawName = normalizedText(model?.name);
  const rawCode = normalizedText(model?.code);
  const name = rawName || rawCode || fallback;

  return {
    name,
    code: rawCode && rawCode !== name ? rawCode : undefined,
  };
}

export function getModelDisplayLabel(
  model: ModelDisplaySource | null | undefined,
  fallback?: string,
) {
  const { name, code } = getModelDisplayParts(model, fallback);
  return code ? `${name} · ${code}` : name;
}
