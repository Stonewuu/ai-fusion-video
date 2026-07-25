export interface AssistantPoint {
  x: number;
  y: number;
}

export interface AssistantRect extends AssistantPoint {
  width: number;
  height: number;
}

export type ResizeDirection =
  | "n"
  | "s"
  | "e"
  | "w"
  | "ne"
  | "nw"
  | "se"
  | "sw";

export const ASSISTANT_MIN_WIDTH = 300;
export const ASSISTANT_MIN_HEIGHT = 300;
export const ASSISTANT_DOCK_MIN_WIDTH = 300;
export const ASSISTANT_DEFAULT_WIDTH = 1000;
export const ASSISTANT_VIEWPORT_GAP = 12;
export const ASSISTANT_LAUNCHER_SIZE = 48;

function finite(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function getViewportSize() {
  if (typeof window === "undefined") {
    return { width: 1280, height: 800 };
  }
  return {
    width: Math.max(window.innerWidth, 320),
    height: Math.max(window.innerHeight, 320),
  };
}

export function getDefaultLauncherPosition(): AssistantPoint {
  const viewport = getViewportSize();
  return {
    x: Math.max(16, viewport.width - ASSISTANT_LAUNCHER_SIZE - 16),
    y: Math.max(16, viewport.height - ASSISTANT_LAUNCHER_SIZE - 16),
  };
}

export function getDefaultNormalRect(): AssistantRect {
  const viewport = getViewportSize();
  const maxWidth = Math.max(1, viewport.width - ASSISTANT_VIEWPORT_GAP * 2);
  const maxHeight = Math.max(1, viewport.height - ASSISTANT_VIEWPORT_GAP * 2);
  const preferredWidth = Math.min(640, Math.max(560, viewport.width * 0.45));
  const width = Math.min(
    maxWidth,
    Math.max(Math.min(ASSISTANT_MIN_WIDTH, maxWidth), preferredWidth),
  );
  const height = maxHeight;
  return {
    width,
    height,
    x: Math.max(ASSISTANT_VIEWPORT_GAP, viewport.width - width - ASSISTANT_VIEWPORT_GAP),
    y: ASSISTANT_VIEWPORT_GAP,
  };
}

export function clampLauncherPosition(
  position: AssistantPoint,
  viewport = getViewportSize(),
): AssistantPoint {
  const maxX = Math.max(16, viewport.width - ASSISTANT_LAUNCHER_SIZE - 16);
  const maxY = Math.max(16, viewport.height - ASSISTANT_LAUNCHER_SIZE - 16);
  return {
    x: Math.min(maxX, Math.max(16, finite(position.x, maxX))),
    y: Math.min(maxY, Math.max(16, finite(position.y, maxY))),
  };
}

export function clampRect(
  rect: AssistantRect,
  viewport = getViewportSize(),
): AssistantRect {
  const maxWidth = Math.max(1, viewport.width - ASSISTANT_VIEWPORT_GAP * 2);
  const maxHeight = Math.max(1, viewport.height - ASSISTANT_VIEWPORT_GAP * 2);
  const defaultRect = getDefaultNormalRect();
  const minWidth = Math.min(ASSISTANT_MIN_WIDTH, maxWidth);
  const minHeight = Math.min(ASSISTANT_MIN_HEIGHT, maxHeight);
  const width = Math.min(
    Math.max(minWidth, finite(rect.width, defaultRect.width)),
    maxWidth,
  );
  const height = Math.min(
    Math.max(minHeight, finite(rect.height, defaultRect.height)),
    maxHeight,
  );
  const maxX = Math.max(ASSISTANT_VIEWPORT_GAP, viewport.width - width - ASSISTANT_VIEWPORT_GAP);
  const maxY = Math.max(ASSISTANT_VIEWPORT_GAP, viewport.height - height - ASSISTANT_VIEWPORT_GAP);
  return {
    width,
    height,
    x: Math.min(maxX, Math.max(ASSISTANT_VIEWPORT_GAP, finite(rect.x, maxX))),
    y: Math.min(maxY, Math.max(ASSISTANT_VIEWPORT_GAP, finite(rect.y, maxY))),
  };
}

export function clampDockWidth(
  width: number,
  availableWidth: number,
  minMainWidth = 300,
): number {
  const safeAvailable = Math.max(0, finite(availableWidth, 0));
  const min = Math.max(ASSISTANT_DOCK_MIN_WIDTH, Math.min(300, safeAvailable * 0.25));
  const max = Math.min(
    safeAvailable - Math.max(minMainWidth, 0),
    safeAvailable * 0.8,
  );
  if (max < min) return 0;
  return Math.min(max, Math.max(min, finite(width, safeAvailable * 0.5)));
}

export function resizeRect(
  start: AssistantRect,
  direction: ResizeDirection,
  deltaX: number,
  deltaY: number,
  viewport = getViewportSize(),
): AssistantRect {
  let { x, y, width, height } = start;
  if (direction.includes("e")) width += deltaX;
  if (direction.includes("s")) height += deltaY;
  if (direction.includes("w")) {
    x += deltaX;
    width -= deltaX;
  }
  if (direction.includes("n")) {
    y += deltaY;
    height -= deltaY;
  }

  if (width < ASSISTANT_MIN_WIDTH) {
    if (direction.includes("w")) x -= ASSISTANT_MIN_WIDTH - width;
    width = ASSISTANT_MIN_WIDTH;
  }
  if (height < ASSISTANT_MIN_HEIGHT) {
    if (direction.includes("n")) y -= ASSISTANT_MIN_HEIGHT - height;
    height = ASSISTANT_MIN_HEIGHT;
  }

  return clampRect({ x, y, width, height }, viewport);
}
