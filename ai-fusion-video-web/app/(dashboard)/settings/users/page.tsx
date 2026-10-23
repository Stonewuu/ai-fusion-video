"use client";

import { useEffect, useMemo, useState } from "react";
import { Loader2, Search, ShieldCheck, Users } from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/lib/store/auth-store";
import { userApi } from "@/lib/api/user";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { Button } from "@/components/ui/button";
import type { UserRespVO } from "@/lib/api/types";
import { containerVariants, itemVariants, settingsTypography } from "../_shared";

function formatDateTime(value?: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export default function SettingsUsersPage() {
  const currentUser = useAuthStore((state) => state.user);
  const isAdmin = currentUser?.roles?.includes("admin") ?? false;
  const [keyword, setKeyword] = useState("");
  const [page, setPage] = useState(1);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const normalizedKeyword = keyword.trim();
  const requestKey = `${normalizedKeyword}:${page}:${retry}`;
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [users, setUsers] = useState<UserRespVO[]>([]);
  const [total, setTotal] = useState(0);
  const loading = isAdmin && loadedKey !== requestKey;

  useEffect(() => {
    if (!isAdmin) return;

    let cancelled = false;

    userApi.page({
      pageNo: page,
      pageSize: 20,
      username: normalizedKeyword || undefined,
    })
      .then((result) => {
        if (cancelled) return;
        setUsers(result.list);
        setTotal(result.total);
        setLoadError(null);
      })
      .catch((err) => {
        if (!cancelled) {
          setLoadError(getApiErrorMessage(err));
          setUsers([]);
          setTotal(0);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadedKey(requestKey);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [isAdmin, normalizedKeyword, page, requestKey]);

  const adminCount = useMemo(
    () => users.filter((user) => user.roles.includes("admin")).length,
    [users]
  );

  if (!isAdmin) {
    return (
      <motion.div
        className="w-full"
        variants={containerVariants}
        initial={false}
        animate="visible"
      >
        <motion.div variants={itemVariants} className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-6">
          <h1 className={settingsTypography.pageTitle}>用户列表</h1>
          <p className={settingsTypography.pageDescription}>
            只有管理员可以查看全部用户。普通用户仅可在个人设置中维护自己的资料。
          </p>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="w-full"
      variants={containerVariants}
      initial={false}
      animate="visible"
    >
      <motion.div variants={itemVariants} className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className={settingsTypography.pageTitle}>用户列表</h1>
          <p className={settingsTypography.pageDescription}>
            查看和搜索系统中的用户、角色与启用状态。
          </p>
        </div>

        <div className="relative w-full md:w-[280px]">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground/60" />
          <input
            type="text"
            value={keyword}
            onChange={(e) => { setKeyword(e.target.value); setPage(1); }}
            aria-label="按用户名搜索"
            placeholder="按用户名搜索"
            className="w-full rounded-xl border border-border/30 bg-card/40 py-2.5 pl-10 pr-4 text-sm outline-none transition-colors focus:border-primary/40"
          />
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="grid gap-4 md:grid-cols-3">
        <div className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="h-4 w-4 text-cyan-500" />
            {normalizedKeyword ? "匹配用户" : "用户总数"}
          </div>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{total}</p>
        </div>

        <div className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ShieldCheck className="h-4 w-4 text-emerald-500" />
            本页管理员
          </div>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{adminCount}</p>
        </div>

        <div className="rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm p-5">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="h-4 w-4 text-blue-500" />
            本页普通成员
          </div>
          <p className="mt-3 text-3xl font-semibold tracking-tight">{users.length - adminCount}</p>
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="mt-6 rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : loadError ? (
          <div className="flex flex-col items-center gap-3 p-6">
            <p role="alert" className="text-sm text-destructive">用户列表加载失败：{loadError}</p>
            <Button variant="outline" onClick={() => setRetry(value => value + 1)}>重试</Button>
          </div>
        ) : users.length === 0 ? (
          <div className="py-16 text-center text-sm text-muted-foreground">
            暂无匹配的用户
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-muted/20 text-muted-foreground">
                <tr>
                  <th className="px-5 py-3 text-left font-medium">用户名</th>
                  <th className="px-5 py-3 text-left font-medium">昵称</th>
                  <th className="px-5 py-3 text-left font-medium">角色</th>
                  <th className="px-5 py-3 text-left font-medium">状态</th>
                  <th className="px-5 py-3 text-left font-medium">注册时间</th>
                </tr>
              </thead>
              <tbody>
                {users.map((user) => {
                  const isUserAdmin = user.roles.includes("admin");
                  return (
                    <tr key={user.id} className="border-t border-border/20">
                      <td className="px-5 py-4 font-medium">{user.username}</td>
                      <td className="px-5 py-4 text-muted-foreground">{user.nickname || "--"}</td>
                      <td className="px-5 py-4">
                        <span
                          className={cn(
                            "inline-flex rounded-full px-2.5 py-1 text-xs font-medium border",
                            isUserAdmin
                              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-600"
                              : "border-border/30 bg-muted/20 text-foreground/80"
                          )}
                        >
                          {isUserAdmin ? "管理员" : "成员"}
                        </span>
                      </td>
                      <td className="px-5 py-4">
                        <span
                          className={cn(
                            "inline-flex rounded-full px-2.5 py-1 text-xs font-medium border",
                            user.status === 1
                              ? "border-blue-500/30 bg-blue-500/10 text-blue-600"
                              : "border-rose-500/30 bg-rose-500/10 text-rose-600"
                          )}
                        >
                          {user.status === 1 ? "启用" : "禁用"}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-muted-foreground">{formatDateTime(user.createTime)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </motion.div>
      {!loading && !loadError && total > 20 && (
        <nav aria-label="用户列表分页" className="mt-4 flex items-center justify-between gap-4">
          <p className="text-sm text-muted-foreground">第 {page} / {Math.ceil(total / 20)} 页 · 共 {total} 人</p>
          <div className="flex gap-2">
            <Button variant="outline" disabled={page === 1} onClick={() => setPage(value => value - 1)}>上一页</Button>
            <Button variant="outline" disabled={page * 20 >= total} onClick={() => setPage(value => value + 1)}>下一页</Button>
          </div>
        </nav>
      )}
    </motion.div>
  );
}
