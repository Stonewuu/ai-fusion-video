"use client";

import AssetEditPanel from "./asset-detail/asset-detail-panel";
import { AssetCreatePanel } from "./asset-detail/asset-create-panel";
import type { Props } from "./asset-detail/shared";

export default function AssetDetailPanel(props: Props) {
  return props.isCreating ? (
    <AssetCreatePanel {...props} />
  ) : (
    <AssetEditPanel key={props.asset.id} {...props} />
  );
}
