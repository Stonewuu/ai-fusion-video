"use client";

import {
  createContext,
  useContext,
  useState,
  type ReactElement,
  type ReactNode,
} from "react";
import { Tooltip } from "@base-ui/react/tooltip";
import { FixedTooltipContent } from "./fixed-tooltip-content";

const DirectoryTooltipContext = createContext<Tooltip.Handle<string> | null>(
  null,
);

export function DirectoryTooltipScope({ children }: { children: ReactNode }) {
  const [handle] = useState(() => Tooltip.createHandle<string>());
  return (
    <DirectoryTooltipContext.Provider value={handle}>
      {children}
      <Tooltip.Root handle={handle} disableHoverablePopup>
        {({ payload }) => (
          <FixedTooltipContent
            data-slot="directory-tooltip"
            side="top"
            align="start"
            sideOffset={8}
          >
            {payload}
          </FixedTooltipContent>
        )}
      </Tooltip.Root>
    </DirectoryTooltipContext.Provider>
  );
}

export function DirectoryTooltipTrigger({
  text,
  render,
}: {
  text: string;
  render: ReactElement;
}) {
  const handle = useContext(DirectoryTooltipContext)!;
  return (
    <Tooltip.Trigger handle={handle} payload={text} render={render} delay={0} />
  );
}
