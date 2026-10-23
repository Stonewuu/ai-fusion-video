"use client";
import { useParams } from "next/navigation";
import { CanvasWorkspace } from "@/components/canvas/canvas-workspace";

export default function PersonalCanvasPage() {
  const { canvasId } = useParams<{ canvasId: string }>();
  return <CanvasWorkspace canvasId={canvasId} />;
}
