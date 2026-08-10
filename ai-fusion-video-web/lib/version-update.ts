const IGNORED_VERSION_KEY = "system-version-ignored-version";
const SESSION_TOAST_PREFIX = "system-version-toast:";

function canUseBrowserStorage(): boolean {
  return typeof window !== "undefined";
}

export function getIgnoredVersion(): string | null {
  if (!canUseBrowserStorage()) return null;
  return localStorage.getItem(IGNORED_VERSION_KEY);
}

export function setIgnoredVersion(version: string): void {
  if (!canUseBrowserStorage()) return;
  localStorage.setItem(IGNORED_VERSION_KEY, version);
}

export function clearIgnoredVersion(): void {
  if (!canUseBrowserStorage()) return;
  localStorage.removeItem(IGNORED_VERSION_KEY);
}

export function markVersionToastShown(version: string): boolean {
  if (!canUseBrowserStorage()) return false;
  const key = `${SESSION_TOAST_PREFIX}${version}`;
  if (sessionStorage.getItem(key) === "1") {
    return false;
  }
  sessionStorage.setItem(key, "1");
  return true;
}