"use client";
import { useParams } from "next/navigation";
import { CanvasWorkspace } from "@/components/canvas/canvas-workspace";

export default function ProjectCanvasPage() {
  const { id, canvasId } = useParams<{ id: string; canvasId: string }>();
  return <CanvasWorkspace canvasId={canvasId} projectId={Number(id)} />;
}
