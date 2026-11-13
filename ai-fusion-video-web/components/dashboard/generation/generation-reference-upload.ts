import { uploadMedia } from "@/lib/api/media";
import type { SimpleAttachmentMediaType } from "./generation-types";

const MEDIA_LABELS = { image: "图片", video: "视频", audio: "音频" };

export async function uploadGenerationReference(file: File, kind: SimpleAttachmentMediaType): Promise<string> {
  const media = await uploadMedia(file, { requestId: crypto.randomUUID() });
  if (media.content.kind.toLowerCase() !== kind) {
    throw new Error(`请选择${MEDIA_LABELS[kind]}文件`);
  }
  return `media://${media.id}`;
}
