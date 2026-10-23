import { http } from "./client";

// ========== 类型定义 ==========

export interface ArtStylePreset {
  key: string;
  name: string;
  description: string;
  imagePrompt: string;
  referenceImagePath: string;
  /** 已登记的全局预设引用或历史公开 URL */
  referenceImagePublicUrl: string | null;
}

// ========== API ==========

export const artStyleApi = {
  publish: (key: string, requestId: string) => http.post<never, string>(`/api/media-presets/${encodeURIComponent(key)}/publish`, { requestId }),
  /** 获取预设画风列表 */
  getPresets: () =>
    http.get<never, ArtStylePreset[]>("/api/project/presets/art-styles"),
};
