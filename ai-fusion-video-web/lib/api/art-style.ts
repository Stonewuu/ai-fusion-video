import { http } from "./client";

// ========== 类型定义 ==========

export interface ArtStylePreset {
  key: string;
  name: string;
  description: string;
  imagePrompt: string;
  referenceImagePath: string;
  /** 上传到 OSS 后的公网 URL（全局） */
  referenceImagePublicUrl: string | null;
}

// ========== API ==========

export const artStyleApi = {
  /** 获取预设画风列表 */
  getPresets: () =>
    http.get<never, ArtStylePreset[]>("/api/project/presets/art-styles"),
};
