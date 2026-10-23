"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useRef } from "react";
import { flushSync } from "react-dom";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const animating = useRef(false);

  const toggle = async (button: HTMLButtonElement) => {
    if (animating.current) return;
    const next = resolvedTheme === "dark" ? "light" : "dark";
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setTheme(next);
      return;
    }
    const { left, top, width, height } = button.getBoundingClientRect();
    const x = left + width / 2;
    const y = top + height / 2;
    const radius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y),
    );
    animating.current = true;
    try {
      const transition = document.startViewTransition(() => {
        // The same provider updates the document, the canvas and every subscriber
        // before the browser captures the new theme for the circular reveal.
        flushSync(() => setTheme(next));
      });
      await transition.ready.then(
        () => {
          document.documentElement.animate(
            {
              clipPath: [
                `circle(0px at ${x}px ${y}px)`,
                `circle(${radius}px at ${x}px ${y}px)`,
              ],
            },
            {
              duration: 450,
              easing: "ease-out",
              pseudoElement: "::view-transition-new(root)",
            },
          );
        },
        () => {
          // A skipped transition (for example, a hidden tab) has no visual frame.
          // Its update callback remains responsible for the single theme change.
        },
      );
      await transition.finished;
    } finally {
      animating.current = false;
    }
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="切换主题"
      title="切换深色 / 浅色模式"
      onClick={(event) => void toggle(event.currentTarget)}
    >
      <Sun className="hidden dark:block" />
      <Moon className="dark:hidden" />
    </Button>
  );
}
