import { http } from "./client";

export interface SystemRuntimeVersionInfo {
  currentVersion: string;
  currentVersionDisplay: string;
  comparisonEnabled: boolean;
  developmentBuild: boolean;
  buildProfile: string;
  message: string | null;
}

export interface SystemVersionInfo {
  currentVersion: string;
  currentVersionDisplay: string;
  latestVersion: string | null;
  latestVersionDisplay: string;
  latestReleaseVersion: string | null;
  latestReleaseVersionDisplay: string;
  latestReleaseUrl: string;
  latestReleasePublishedAt: string | null;
  latestReleaseDockerReady: boolean;
  updateAvailable: boolean;
  comparisonEnabled: boolean;
  developmentBuild: boolean;
  buildProfile: string;
  versionRelation: "behind" | "same" | "ahead" | "incomparable";
  releaseUrl: string;
  tagUrl: string;
  publishedAt: string | null;
  checkedAt: string;
  source: string;
  checkSucceeded: boolean;
  message: string | null;
}

export function getSystemRuntimeVersion(): Promise<SystemRuntimeVersionInfo> {
  return http.get("/api/system/version/runtime");
}

export function getSystemVersion(force = false): Promise<SystemVersionInfo> {
  return http.get("/api/system/version", {
    params: force ? { force: true } : undefined,
  });
}