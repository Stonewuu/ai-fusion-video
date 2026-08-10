import type { Metadata } from "next";
import { MainContentFrame } from "@/components/dashboard/main-content-frame";

export const metadata: Metadata = {
  title: "素材",
};

export default function AssetsLayout({ children }: { children: React.ReactNode }) {
  return <MainContentFrame>{children}</MainContentFrame>;
}
