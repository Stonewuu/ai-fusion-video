"use client";
import { createContext, useContext, type ReactNode } from "react";
import { useMediaImportsController } from "./use-media-imports";

const Context = createContext<ReturnType<
  typeof useMediaImportsController
> | null>(null);
export function CanvasMediaImportsProvider({
  userId,
  children,
}: {
  userId: number;
  children: ReactNode;
}) {
  const value = useMediaImportsController(userId);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}
export function useCanvasMediaImports() {
  const value = useContext(Context);
  if (!value) throw new Error("导入组件必须位于画布导入工作区内");
  return value;
}
