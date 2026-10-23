"use client";

import { Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAuthStore } from "@/lib/store/auth-store";
import { DefinitionsSection } from "./_components/definitions-section";
import { ConfigSectionLoader } from "./_components/config-section-loader";
import { settingsTypography } from "../_shared";

const sections = [{ value: "definitions", label: "智能体" }, { value: "skills", label: "Skills" }, { value: "mcp", label: "MCP" }, { value: "runtime", label: "运行与存储" }];

function AgentSettingsContent() {
  const user = useAuthStore((state) => state.user);
  const router = useRouter();
  const search = useSearchParams();
  const requested = search.get("section");
  const section = sections.some((item) => item.value === requested) ? requested! : "definitions";
  return <div className="space-y-6">
    <header><h1 className={settingsTypography.pageTitle}>智能体配置</h1><p className={settingsTypography.pageDescription}>调整你的智能体，管理工作规范与外部工具。</p></header>
    <Tabs value={section} onValueChange={(value) => router.replace(`/settings/agents?section=${value}`, { scroll: false })} className="gap-6">
      <div className="overflow-x-auto"><TabsList aria-label="智能体设置分类">{sections.map((item) => <TabsTrigger key={item.value} value={item.value} className="transition-colors motion-reduce:transition-none" onFocus={(event) => event.currentTarget.scrollIntoView({ block: "nearest", inline: "center" })}>{item.label}</TabsTrigger>)}</TabsList></div>
      {user && <>
        <TabsContent value="definitions"><DefinitionsSection key={user.id} userId={user.id} /></TabsContent>
        {sections.slice(1).map((item) => <TabsContent key={item.value} value={item.value}><ConfigSectionLoader key={`${user.id}-${item.value}`} section={item.value} userId={user.id} isAdmin={user.roles?.includes("admin") ?? false} /></TabsContent>)}
      </>}
    </Tabs>
  </div>;
}

export default function AgentSettingsPage() {
  return <Suspense fallback={<div role="status" className="flex items-center gap-2 p-6 text-muted-foreground"><Loader2 className="size-4 animate-spin" />加载配置中…</div>}><AgentSettingsContent /></Suspense>;
}
