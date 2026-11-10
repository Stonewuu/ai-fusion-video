"use client";

import { useEffect, useRef } from "react";
import { usePageVisible } from "@/hooks/use-page-visible";
import { usePrefersReducedMotion } from "@/hooks/use-prefers-reduced-motion";
import { createAuthDotRenderer } from "./auth-dot-renderer";

export function AuthAnimatedBackground({ paused }: { paused: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<ReturnType<typeof createAuthDotRenderer> | null>(null);
  const pageVisible = usePageVisible();
  const reducedMotion = usePrefersReducedMotion();
  const animating = pageVisible && !reducedMotion && !paused;

  useEffect(() => {
    const canvas = canvasRef.current!;
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const renderer = createAuthDotRenderer(canvas, context);
    rendererRef.current = renderer;
    const resize = () => {
      renderer.resize(canvas.clientWidth, canvas.clientHeight);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    return () => {
      observer.disconnect();
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    rendererRef.current?.setAnimating(animating);
  }, [animating]);

  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden [contain:strict]">
      <canvas ref={canvasRef} data-auth-background data-animating={animating} className="h-full w-full" />
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(ellipse at center, var(--background) 5%, color-mix(in oklab, var(--background) 85%, transparent) 35%, transparent 100%)",
        }}
      />
    </div>
  );
}
