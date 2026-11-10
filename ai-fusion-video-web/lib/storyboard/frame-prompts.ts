import type { Project } from "@/lib/api/project";
import type { StoryboardFrameType, StoryboardItem } from "@/lib/api/storyboard";

const FRAME_INSTRUCTIONS: Record<StoryboardFrameType, string> = {
  first: "生成本镜头开始时的一个静态画面，为主要动作留下变化空间；延续上一镜头已成立的状态，不提前完成后续动作。持续行走等动作可从运动中的瞬间开始。",
  last: "生成本镜头结束时的一个静态画面，表现本镜头动作推进后的状态，不提前进入下一镜头。有本镜头首帧时，保持人物、场景、道具、光照与空间关系一致。",
};

const CONTINUITY_INSTRUCTION = "先按场次统一人物外貌、服饰与状态、地点、时间、天气、道具形状与归属，再逐镜头确定构图。遵循景别和机位，画面只出现当前可见的主体；对白用于理解情绪，不画成字幕。参考图只继承指定特征，不增加无来源的布景、人物或道具。";

export function buildDefaultFramePrompt(
  item: StoryboardItem,
  project: Project | null | undefined,
  frameType: StoryboardFrameType,
) {
  const savedPrompt = frameType === "first" ? item.firstFramePrompt : item.lastFramePrompt;
  if (savedPrompt?.trim()) {
    return savedPrompt;
  }
  const style = [project?.artStyleDescription, project?.artStyleImagePrompt, project?.artStyle]
    .find(text => text?.trim());
  const parts = [
    FRAME_INSTRUCTIONS[frameType],
    CONTINUITY_INSTRUCTION,
    style ? `项目画风：${style}` : null,
    `镜头：${item.shotNumber || item.autoShotNumber || item.id}`,
    item.shotType ? `景别：${item.shotType}` : null,
    item.cameraAngle ? `机位角度：${item.cameraAngle}` : null,
    item.content ? `镜头完整动作（用于确定起止状态）：${item.content}` : null,
    item.sceneExpectation ? `画面期望：${item.sceneExpectation}` : null,
    item.dialogue ? `对白/旁白（仅供理解）：${item.dialogue}` : null,
    item.cameraMovement ? `视频运镜（取对应起点或终点构图）：${item.cameraMovement}` : null,
  ];
  return parts.filter(Boolean).join("\n");
}

export function buildDefaultBatchFramePrompt(frameType: StoryboardFrameType) {
  return `${FRAME_INSTRUCTIONS[frameType]}\n${CONTINUITY_INSTRUCTION}`;
}

/** 同场首尾帧交给一次调度，避免两个工作流并发改写同一镜头。 */
export function buildBatchFrameContext(storyboardId: number, firstItemIds: number[], lastItemIds: number[]) {
  const frameRequests = [
    { frameType: "first" as const, storyboardItemIds: [...new Set(firstItemIds)] },
    { frameType: "last" as const, storyboardItemIds: [...new Set(lastItemIds)] },
  ].filter(request => request.storyboardItemIds.length > 0);
  if (frameRequests.length === 0) {
    return null;
  }
  return {
    storyboardId,
    selectedStoryboardItemIds: [...new Set(frameRequests.flatMap(request => request.storyboardItemIds))],
    frameType: frameRequests.length === 2 ? "both" : frameRequests[0].frameType,
    frameRequests,
    framePrompt: frameRequests.map(request => `${request.frameType === "first" ? "首帧" : "尾帧"}：${buildDefaultBatchFramePrompt(request.frameType)}`).join("\n\n"),
  };
}
