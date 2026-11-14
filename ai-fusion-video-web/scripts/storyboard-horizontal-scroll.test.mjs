import { readFile } from "node:fs/promises";
import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL(
  "../app/(dashboard)/projects/[id]/storyboards/_components/storyboard-horizontal-scroll.ts", import.meta.url,
), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 },
});
const { attachStoryboardHorizontalScroll } = await import(
  "data:text/javascript;base64," + Buffer.from(outputText).toString("base64")
);

class Viewport extends EventTarget {
  style = {};
  clientWidth = 400;
  limit = 800;
  position = 0;
  get scrollLeft() {
    return this.position;
  }
  set scrollLeft(value) {
    this.position = Math.max(0, Math.min(value, this.limit));
  }
}

function fixture(t) {
  const observer = { disconnected: false };
  const previous = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class {
    constructor(callback) {
      observer.notify = callback;
    }
    observe() {
    }
    disconnect() {
      observer.disconnected = true;
    }
  };
  t.after(() => {
    globalThis.ResizeObserver = previous;
  });
  const header = new Viewport();
  const content = new Viewport();
  const body = new Viewport();
  const detach = attachStoryboardHorizontalScroll(header, content, body);
  t.after(detach);
  return { header, content, body, detach, observer };
}

function wheel(target, deltaX, deltaY = 0, shiftKey = false, deltaMode = 0) {
  const event = Object.assign(new Event("wheel", { cancelable: true }), {
    deltaX, deltaY, shiftKey, deltaMode,
  });
  target.dispatchEvent(event);
  return event;
}

test("快速反向滚动始终显示表体最新位置，延迟的表头事件不能拉回表体", (t) => {
  const { header, content, body } = fixture(t);
  for (const position of [180, 390, 270, 720, 81, 0]) {
    body.scrollLeft = position;
    body.dispatchEvent(new Event("scroll"));
    header.scrollLeft = 25;
    header.dispatchEvent(new Event("scroll"));
    assert.equal(body.scrollLeft, position);
    assert.equal(content.style.transform, `translateX(${-position}px)`);
  }
});

test("表头接受横向及 Shift 滚轮，纵向滚轮和边界仍交给页面", (t) => {
  const { header, content, body } = fixture(t);
  assert.equal(wheel(header, 120).defaultPrevented, true);
  assert.equal(body.scrollLeft, 120);
  assert.equal(wheel(header, 0, 4, true, 1).defaultPrevented, true);
  assert.equal(body.scrollLeft, 184);
  assert.equal(content.style.transform, "translateX(-184px)");
  assert.equal(wheel(header, 0, 120).defaultPrevented, false);
  assert.equal(body.scrollLeft, 184);
  wheel(header, 1000);
  assert.equal(body.scrollLeft, 800);
  assert.equal(wheel(header, 50).defaultPrevented, false);
});

test("列宽和视口变化后保持对齐，卸载后停止响应", (t) => {
  const { header, content, body, observer, detach } = fixture(t);
  body.scrollLeft = 500;
  observer.notify();
  assert.equal(content.style.transform, "translateX(-500px)");
  body.limit = 200;
  body.scrollLeft = 500;
  observer.notify();
  assert.equal(content.style.transform, "translateX(-200px)");
  detach();
  assert.equal(observer.disconnected, true);
  wheel(header, -100);
  assert.equal(body.scrollLeft, 200);
  body.scrollLeft = 20;
  body.dispatchEvent(new Event("scroll"));
  assert.equal(content.style.transform, "translateX(-200px)");
});
