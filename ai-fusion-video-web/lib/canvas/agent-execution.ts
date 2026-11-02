import type { CanvasAgentExecution } from "@/lib/api/canvas-agents";
import { generationTerminalStates } from "@/lib/generation-execution-view";

export function canvasAgentExecutionFinished(execution: CanvasAgentExecution) {
  return (
    execution.tasks.every((task) =>
      ["COMPLETED", "FAILED", "CANCELLED"].includes(task.status),
    ) &&
    execution.generations.every((generation) =>
      generationTerminalStates.has(generation.state),
    )
  );
}
