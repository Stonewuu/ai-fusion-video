"use client";
import { useEffect, useRef } from "react";
import { registerCanvasAssistantDraft } from "@/lib/canvas/assistant-session";

export function useCanvasAssistantDraft(key: string, save: () => Promise<void>) {
  const latest = useRef(save);
  useEffect(() => { latest.current = save; }, [save]);
  useEffect(() => registerCanvasAssistantDraft(key, () => latest.current()), [key]);
}
