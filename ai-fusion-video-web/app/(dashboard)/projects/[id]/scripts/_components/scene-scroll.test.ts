import assert from "node:assert/strict";
import test from "node:test";
import { animateSceneScroll, SCENE_SCROLL_DURATION_MS, sceneScrollTop } from "./scene-scroll";

function frameClock(run: (step: (now: number) => void) => void) {
  const request = globalThis.requestAnimationFrame;
  const cancel = globalThis.cancelAnimationFrame;
  const frames = new Map<number, FrameRequestCallback>();
  let id = 0;
  globalThis.requestAnimationFrame = (callback) => {
    frames.set(++id, callback);
    return id;
  };
  globalThis.cancelAnimationFrame = (key) => {
    frames.delete(key);
  };
  try {
    run((now) => {
      const callbacks = [...frames.values()];
      frames.clear();
      callbacks.forEach((callback) => callback(now));
    });
  } finally {
    globalThis.requestAnimationFrame = request;
    globalThis.cancelAnimationFrame = cancel;
  }
}

function viewport() {
  const root = {
    scrollTop: 400, scrollHeight: 5000, clientHeight: 800, clientTop: 2,
    getBoundingClientRect: () => ({ top: 100 }), contains: () => true,
  } as unknown as HTMLElement;
  let contentTop = 1240;
  const scene = { getBoundingClientRect: () => ({ top: 102 + contentTop - root.scrollTop, height: 2200 }) } as HTMLElement;
  return { root, scene, moveScene: (top: number) => {
    contentTop = top;
  } };
}

test("长场次定位到标题上方的安全间距，而不是场次正文中间", () => {
  const { root, scene, moveScene } = viewport();
  assert.equal(sceneScrollTop(root, scene), 1228);
  moveScene(4990);
  assert.equal(sceneScrollTop(root, scene), 4200);
});

test("滚动持续 0.5 秒，结束时根据当前布局精确对齐标题", () => {
  frameClock((step) => {
    const { root, scene, moveScene } = viewport();
    let completed = false;
    animateSceneScroll(root, scene, SCENE_SCROLL_DURATION_MS, () => {
      completed = true;
    });
    step(0);
    step(250);
    assert.ok(root.scrollTop > 400 && root.scrollTop < 1228);
    assert.equal(completed, false);
    moveScene(1500);
    step(499);
    assert.equal(completed, false);
    step(500);
    assert.equal(root.scrollTop, 1488);
    assert.equal(completed, true);
  });
});

test("连续选择会取消旧动画，不再被旧目标拉回", () => {
  frameClock((step) => {
    const { root, scene, moveScene } = viewport();
    const stop = animateSceneScroll(root, scene, 500, () => assert.fail("旧动画已被取消"));
    step(0);
    step(100);
    stop();
    const before = root.scrollTop;
    step(500);
    assert.equal(root.scrollTop, before);
    moveScene(2400);
    let completed = false;
    animateSceneScroll(root, scene, 500, () => {
      completed = true;
    });
    step(600);
    step(1100);
    assert.equal(root.scrollTop, 2388);
    assert.equal(completed, true);
  });
});

test("减少动态效果时直接到达相同位置", () => {
  frameClock((step) => {
    const { root, scene } = viewport();
    let completed = false;
    animateSceneScroll(root, scene, 0, () => {
      completed = true;
    });
    step(0);
    assert.equal(root.scrollTop, 1228);
    assert.equal(completed, true);
  });
});
