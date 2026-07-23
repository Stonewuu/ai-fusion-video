import { http } from "./client";
import type { PageResult } from "./types";

export interface GenerationTaskBase {
  id: number;
  taskId: string;
  prompt: string;
  modelId: number | null;
  count: number;
  successCount: number;
  status: number;
  errorMsg: string | null;
  createTime: string;
  updateTime: string;
}

export interface ImageGenerationTask extends GenerationTaskBase {
  projectId: number | null;
  refImageUrls: string | null;
  ratio: string | null;
  resolution: string | null;
  aspectRatio: string | null;
  width: number | null;
  height: number | null;
}

export interface VideoGenerationTask extends GenerationTaskBase {
  projectId: number | null;
  generateMode: string | null;
  firstFrameImageUrl: string | null;
  lastFrameImageUrl: string | null;
  referenceImageUrls: string | null;
  referenceVideoUrls: string | null;
  referenceAudioUrls: string | null;
  ratio: string | null;
  resolution: string | null;
  duration: number | null;
  watermark: boolean | null;
  generateAudio: boolean | null;
  seed: number | null;
  cameraFixed: boolean | null;
}

export interface ImageGenerationItem {
  id: number;
  taskId: number;
  imageUrl: string | null;
  thumbnailUrl: string | null;
  width: number | null;
  height: number | null;
  status: number;
  errorMsg: string | null;
}

export interface VideoGenerationItem {
  id: number;
  taskId: number;
  videoUrl: string | null;
  coverUrl: string | null;
  duration: number | null;
  status: number;
  errorMsg: string | null;
  firstFrameUrl: string | null;
  lastFrameUrl: string | null;
}

export interface ImageGenerationSubmitReq {
  projectId?: number;
  prompt: string;
  refImageUrls?: string;
  ratio?: string;
  resolution?: string;
  aspectRatio?: string;
  width?: number;
  height?: number;
  count?: number;
  modelId: number;
  category?: string;
}

export interface VideoGenerationSubmitReq {
  projectId?: number;
  prompt: string;
  generateMode?: string;
  firstFrameImageUrl?: string;
  lastFrameImageUrl?: string;
  referenceImageUrls?: string;
  referenceVideoUrls?: string;
  referenceAudioUrls?: string;
  ratio?: string;
  resolution?: string;
  duration?: number;
  watermark?: boolean;
  generateAudio?: boolean;
  seed?: number;
  cameraFixed?: boolean;
  count?: number;
  modelId: number;
  category?: string;
}

export const generationApi = {
  submitImage: (data: ImageGenerationSubmitReq) =>
    http.post<never, string>("/api/generation/image/submit", data),
  getImageTask: (taskId: string) =>
    http.get<never, ImageGenerationTask>(`/api/generation/image/${encodeURIComponent(taskId)}`),
  listImageItems: (taskId: number) =>
    http.get<never, ImageGenerationItem[]>(`/api/generation/image/${taskId}/items`),
  pageImageTasks: (pageNo = 1, pageSize = 10) =>
    http.get<never, PageResult<ImageGenerationTask>>(
      `/api/generation/image/page?pageNo=${pageNo}&pageSize=${pageSize}`,
    ),

  submitVideo: (data: VideoGenerationSubmitReq) =>
    http.post<never, string>("/api/generation/video/submit", data),
  getVideoTask: (taskId: string) =>
    http.get<never, VideoGenerationTask>(`/api/generation/video/${encodeURIComponent(taskId)}`),
  listVideoItems: (taskId: number) =>
    http.get<never, VideoGenerationItem[]>(`/api/generation/video/${taskId}/items`),
  pageVideoTasks: (pageNo = 1, pageSize = 10) =>
    http.get<never, PageResult<VideoGenerationTask>>(
      `/api/generation/video/page?pageNo=${pageNo}&pageSize=${pageSize}`,
    ),
};
