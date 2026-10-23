import { canvasPortLabels } from "@/lib/canvas/labels";
import type { CanvasGraph } from "@/lib/canvas/types";

export function CanvasPlanInputs({
  nodeId,
  dependencies,
  graph,
}: {
  nodeId: string;
  dependencies: string[];
  graph: CanvasGraph;
}) {
  const titles = new Map(graph.nodes.map((node) => [node.id, node.title]));
  const incoming = graph.edges
    .filter((edge) => edge.kind === "execution" && edge.targetNodeId === nodeId)
    .sort(
      (a, b) =>
        a.targetPort!.localeCompare(b.targetPort!) ||
        a.inputIndex! - b.inputIndex!,
    );
  if (!incoming.length) return null;
  return (
    <div className="mt-2 flex flex-col gap-1 text-xs text-muted-foreground">
      <p>
        {dependencies.length
          ? `等待：${dependencies.map((id) => titles.get(id)).join("、")}`
          : "使用已固定的内容"}
      </p>
      <ul className="flex flex-col gap-1" aria-label="本步骤的输入用途">
        {incoming.map((edge) => (
          <li key={edge.id}>
            {titles.get(edge.sourceNodeId)} →{" "}
            {canvasPortLabels[edge.targetPort!] ?? edge.targetPort}
            {edge.output ? "（固定结果）" : ""}
          </li>
        ))}
      </ul>
    </div>
  );
}
