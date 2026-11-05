export const SCENE_SCROLL_DURATION_MS = 500;
const SCENE_TOP_GAP = 12;

export function sceneScrollTop(container: HTMLElement, scene: HTMLElement): number {
  const top = container.scrollTop + scene.getBoundingClientRect().top
    - container.getBoundingClientRect().top - container.clientTop - SCENE_TOP_GAP;
  return Math.max(0, Math.min(top, container.scrollHeight - container.clientHeight));
}

export function animateSceneScroll(container: HTMLElement, scene: HTMLElement, duration: number, onComplete: () => void) {
  const from = container.scrollTop;
  let startedAt: number | undefined;
  let frame = 0;
  let cancelled = false;
  const tick = (now: number) => {
    if (cancelled) {
      return;
    }
    if (!container.contains(scene)) {
      onComplete();
      return;
    }
    startedAt ??= now;
    const progress = duration === 0 ? 1 : Math.min((now - startedAt) / duration, 1);
    const eased = 1 - (1 - progress) ** 3;
    // 同一帧重新计算标题位置，避免内容渲染改变卡片上方高度后仍使用旧坐标。
    const target = sceneScrollTop(container, scene);
    container.scrollTop = from + (target - from) * eased;
    if (progress < 1) {
      frame = requestAnimationFrame(tick);
    } else {
      onComplete();
    }
  };
  frame = requestAnimationFrame(tick);
  return () => {
    cancelled = true;
    cancelAnimationFrame(frame);
  };
}
