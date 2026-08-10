const TOOL_RESOURCE_LABELS: Readonly<
  Record<string, Readonly<Record<string, string>>>
> = {
  delete_asset_resource: {
    asset: "主资产",
    item: "子资产",
  },
  delete_script_child_resource: {
    episode: "剧本分集",
    scene: "剧本场次",
  },
  delete_storyboard_child_resource: {
    episode: "分镜集",
    item: "分镜镜头",
    scene: "分镜场次",
  },
};

export function getToolResourceLabel(
  toolName: string | undefined,
  resourceType: string | undefined,
): string | undefined {
  if (!toolName || !resourceType) return undefined;
  return TOOL_RESOURCE_LABELS[toolName]?.[resourceType];
}
