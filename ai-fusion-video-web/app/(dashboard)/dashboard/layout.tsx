import type { Metadata } from "next";
import { MainContentFrame } from "@/components/dashboard/main-content-frame";

export const metadata: Metadata = {
  title: "仪表盘",
};

export default function DashboardPageLayout({ children }: { children: React.ReactNode }) {
  return <MainContentFrame>{children}</MainContentFrame>;
}
