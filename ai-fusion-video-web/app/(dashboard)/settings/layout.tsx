import type { Metadata } from "next";
import { MainContentFrame } from "@/components/dashboard/main-content-frame";

export const metadata: Metadata = {
  title: "设置",
};

export default function SettingsLayout({ children }: { children: React.ReactNode }) {
  return <MainContentFrame width="settings">{children}</MainContentFrame>;
}
