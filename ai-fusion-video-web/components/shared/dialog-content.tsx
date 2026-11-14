"use client";

import type { ComponentProps } from "react";
import {
  DialogContent as BaseDialogContent,
  DialogOverlay,
} from "@/components/ui/dialog";

/** 嵌套弹窗使用自己的遮罩，由 Base UI 处理最上层关闭与焦点恢复。 */
export function DialogContent(
  props: Omit<ComponentProps<typeof BaseDialogContent>, "render">,
) {
  return (
    <BaseDialogContent
      {...props}
      render={(popupProps, state) => (
        <>
          {state.nested && (
            <DialogOverlay forceRender data-modal-layer="nested" />
          )}
          <div {...popupProps} />
        </>
      )}
    />
  );
}
