"use client";

import { useEffect, useRef } from "react";
import { useResourceChangeStore } from "@/lib/store/resource-change-store";
import { createStoryboardRefreshCoordinator } from "@/lib/storyboard-refresh";

export function useStoryboardInvalidation(projectId: number, refresh: () => Promise<void>) {
  const refreshRef = useRef(refresh);
  useEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);

  useEffect(() => {
    const coordinator = createStoryboardRefreshCoordinator(() => refreshRef.current());
    const unsubscribe = useResourceChangeStore.subscribe((state, previous) => {
      coordinator.update(
        state.revisions.storyboards !== previous.revisions.storyboards,
        false,
      );
    });
    return () => {
      unsubscribe();
      coordinator.dispose();
    };
  }, [projectId]);
}
