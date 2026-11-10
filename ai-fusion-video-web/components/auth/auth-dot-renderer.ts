interface Dot {
  x: number;
  y: number;
  seed: number;
}

const STEP_MS = 100;
const PHASES = 50;
const SPACING = 10;
const DOT_SIZE = 3;
const COLORS = ["40,60,140", "70,100,180"];
const OPACITIES = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1];

export function createAuthDotRenderer(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) {
  let phases: Dot[][] = [];
  let phase = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;

  function paint(dot: Dot) {
    dot.seed = (Math.imul(dot.seed, 1664525) + 1013904223) >>> 0;
    const opacity = OPACITIES[dot.seed % OPACITIES.length];
    const color = COLORS[(dot.seed >>> 8) % COLORS.length];
    context.clearRect(dot.x, dot.y, DOT_SIZE, DOT_SIZE);
    context.fillStyle = `rgba(${color},${opacity})`;
    context.fillRect(dot.x, dot.y, DOT_SIZE, DOT_SIZE);
  }

  function tick() {
    // 每个点五秒变化一次，每次只重绘其中一组，避免逐帧清空整张画布。
    for (const dot of phases[phase]) {
      paint(dot);
    }
    phase = (phase + 1) % PHASES;
    timer = setTimeout(tick, STEP_MS);
  }

  function resize(width: number, height: number) {
    if (disposed) {
      return;
    }
    const nextWidth = Math.ceil(width);
    const nextHeight = Math.ceil(height);
    if (canvas.width === nextWidth && canvas.height === nextHeight && phases.length > 0) {
      return;
    }
    // 点阵按 CSS 像素绘制，文字仍由 DOM 渲染，不随高分屏扩大背景纹理。
    canvas.width = nextWidth;
    canvas.height = nextHeight;
    phases = Array.from({ length: PHASES }, () => []);
    for (let y = 0; y < nextHeight; y += SPACING) {
      for (let x = 0; x < nextWidth; x += SPACING) {
        const dot = { x, y, seed: (Math.imul(x + 1, 73856093) ^ Math.imul(y + 1, 19349663)) >>> 0 };
        phases[(dot.seed >>> 16) % PHASES].push(dot);
        paint(dot);
      }
    }
  }

  function setAnimating(active: boolean) {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    if (active && !disposed && phases.length > 0) {
      timer = setTimeout(tick, STEP_MS);
    }
  }

  function dispose() {
    setAnimating(false);
    disposed = true;
    phases = [];
    canvas.width = 0;
    canvas.height = 0;
  }

  return { resize, setAnimating, dispose };
}
