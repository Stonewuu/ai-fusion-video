import type { CanvasResource } from "./types";
import type { CanvasMediaPreview } from "./presentation";

const sourceFields: Record<
  string,
  [string, string, CanvasMediaPreview["kind"]][]
> = {
  project: [
    ["coverUrl", "项目封面", "image"],
    ["artStyleImageUrl", "画风参考", "image"],
  ],
  asset: [["coverUrl", "资产封面", "image"]],
  assetItem: [
    ["imageUrl", "资产图片", "image"],
    ["thumbnailUrl", "缩略图", "image"],
  ],
  shot: [
    ["firstFrameImageUrl", "首帧", "image"],
    ["lastFrameImageUrl", "尾帧", "image"],
    ["videoUrl", "镜头视频", "video"],
    ["generatedVideoUrl", "生成视频", "video"],
    ["referenceImageUrl", "参考图", "image"],
    ["imageUrl", "镜头图片", "image"],
    ["generatedImageUrl", "生成图片", "image"],
    ["customData.videoPosterUrl", "视频封面", "image"],
  ],
  storyboardEpisode: [
    ["composedVideoUrl", "合成视频", "video"],
    ["bgmAudioUrl", "配乐", "audio"],
    ["bgmSfxAudioUrl", "音效", "audio"],
  ],
};
export function resourceMedia(resource: CanvasResource) {
  const fields = [...(sourceFields[resource.header.resourceType] ?? [])];
  if (["asset", "assetItem"].includes(resource.header.resourceType)) {
    fields.push(
      ["properties.videoUrl", "资产视频", "video"],
      ["properties.audioUrl", "资产音频", "audio"],
      ["properties.thumbnailUrl", "素材缩略图", "image"],
    );
    const properties = resource.data.properties as Record<
      string,
      unknown
    > | null;
    const mode = properties?.sourceGenerationMode;
    if (mode === "image" || mode === "video" || mode === "audio")
      fields.push(["properties.sourceMediaUrl", "原始生成素材", mode]);
  }
  return fields.flatMap(([field, title, kind]) => {
    let value: unknown = resource.data;
    for (const part of field.split("."))
      value =
        value && typeof value === "object"
          ? (value as Record<string, unknown>)[part]
          : undefined;
    return typeof value === "string" && value
      ? [{ field, title, kind, source: value }]
      : [];
  });
}

/** 与分集合成一致：人工指定视频优先，否则使用该镜头已保存的生成视频。 */
export function shotVideo(data: Record<string, unknown>) {
  if (typeof data.videoUrl === "string" && data.videoUrl) return data.videoUrl;
  return typeof data.generatedVideoUrl === "string" && data.generatedVideoUrl
    ? data.generatedVideoUrl
    : null;
}
