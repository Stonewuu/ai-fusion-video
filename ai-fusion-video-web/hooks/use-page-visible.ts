"use client";

import { useSyncExternalStore } from "react";

function subscribe(onChange: () => void) {
  document.addEventListener("visibilitychange", onChange);
  return () => {
    document.removeEventListener("visibilitychange", onChange);
  };
}

export function usePageVisible() {
  return useSyncExternalStore(subscribe, () => !document.hidden, () => false);
}
