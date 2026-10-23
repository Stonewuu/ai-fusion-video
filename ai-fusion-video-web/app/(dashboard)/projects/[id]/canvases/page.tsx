"use client";
import { useParams } from "next/navigation";
import { CanvasList } from "@/components/canvas/canvas-list";

export default function ProjectCanvasesPage() {
  const { id } = useParams<{ id: string }>();
  return <CanvasList projectId={Number(id)} />;
}
