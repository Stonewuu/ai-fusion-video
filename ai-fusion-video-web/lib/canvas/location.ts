import type { CanvasDocument } from "./types";
export function canvasLocation(
  document: Pick<CanvasDocument, "id" | "projectId">,
) {
  return document.projectId === null
    ? `/canvases/${document.id}`
    : `/projects/${document.projectId}/canvases/${document.id}`;
}
