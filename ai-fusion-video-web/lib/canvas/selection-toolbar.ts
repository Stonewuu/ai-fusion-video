export function selectionToolbarPosition(
  bounds: {
    left: number;
    top: number;
    right: number;
    bottom: number;
    width: number;
    height: number;
  },
  size: { width: number; height: number },
) {
  const margin = 8;
  const above = bounds.top - size.height - 12;
  const below = bounds.bottom + 12;
  const idealTop =
    above >= margin
      ? above
      : below + size.height <= bounds.height - margin
        ? below
        : above;
  return {
    left: Math.max(
      margin,
      Math.min(
        (bounds.left + bounds.right - size.width) / 2,
        bounds.width - size.width - margin,
      ),
    ),
    top: Math.max(
      margin,
      Math.min(idealTop, bounds.height - size.height - margin),
    ),
  };
}
