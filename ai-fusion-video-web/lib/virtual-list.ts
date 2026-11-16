/** 保留可见项和正在编辑、拖动的项；移除已不在列表中的索引。 */
export function retainVirtualItems(
  visible: number[],
  pinned: readonly number[],
  count: number,
): number[] {
  const indexes = new Set(visible);
  for (const index of pinned) {
    if (index >= 0 && index < count) {
      indexes.add(index);
    }
  }
  return [...indexes].sort((a, b) => a - b);
}
