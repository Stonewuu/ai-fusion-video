import type { ELK } from "elkjs/lib/elk-api.js";
import { buildLayoutScopes } from "./layout/model";
import { layeredPlacement } from "./layout/elk-layout";
import { constrainPlacement } from "./layout/constraints";
import { compactPlacement } from "./layout/compact";
import { measureLayoutCurves } from "./layout/curves";
import { compareQuality, layoutQuality } from "./layout/quality";
import { LayoutConstraintError, type LayoutRequest, type LayoutResult } from "./layout/types";

export type { CanvasGeometry, CanvasNodeGeometry } from "./layout/types";

export async function autoArrangeCanvas(request: LayoutRequest, engine: Pick<ELK, "layout">): Promise<LayoutResult> {
  const scopes = buildLayoutScopes(request);
  const layers = [];
  for (const scope of scopes) {
    layers.push(await layeredPlacement(scope, engine));
  }
  let best: { result: LayoutResult; quality: number[] } | null = null;
  const failures: string[] = [];
  const candidates = new Set<string>();
  for (const balanced of [false, true]) {
    // 每个候选都必须通过相同的固定位置和避让约束，不接受不完整布局。
    try {
      const positions: LayoutResult["positions"] = {};
      for (let index = 0; index < scopes.length; index++) {
        const scope = scopes[index];
        Object.assign(positions, constrainPlacement(scope, compactPlacement(scope, layers[index], balanced)));
      }
      const key = JSON.stringify(positions);
      if (candidates.has(key)) {
        continue;
      }
      candidates.add(key);
      const result = { positions, routes: measureLayoutCurves(request, positions) };
      const quality = layoutQuality(request, result);
      if (!best || compareQuality(quality, best.quality) < 0) {
        best = { result, quality };
      }
    } catch (error) {
      if (!(error instanceof LayoutConstraintError)) {
        throw error;
      }
      failures.push(error.message);
    }
  }
  if (!best) {
    throw new Error([...new Set(failures)].join("；"));
  }
  return best.result;
}

export { autoLayoutOperations } from "./layout/operations";
