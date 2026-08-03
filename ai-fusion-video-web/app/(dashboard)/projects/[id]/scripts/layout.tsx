import { MainContentFrame } from "@/components/dashboard/main-content-frame";

export default function ProjectScriptsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <MainContentFrame fullHeight>{children}</MainContentFrame>;
}
