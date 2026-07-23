import { MainContentFrame } from "@/components/dashboard/main-content-frame";

export default function ProjectAssetsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <MainContentFrame width="full" fullHeight>
      {children}
    </MainContentFrame>
  );
}
