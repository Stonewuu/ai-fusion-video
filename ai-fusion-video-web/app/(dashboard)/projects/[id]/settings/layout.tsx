import { MainContentFrame } from "@/components/dashboard/main-content-frame";

export default function ProjectSettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <MainContentFrame>{children}</MainContentFrame>;
}
