"use client";

import { useState } from "react";
import {
  User,
  Key,
  Loader2,
  Check,
  Mail,
  Phone,
  Eye,
  EyeOff,
} from "lucide-react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/lib/store/auth-store";
import * as authApi from "@/lib/api/auth";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { containerVariants, itemVariants } from "../_shared";

// ============================================================
// 个人信息编辑 Dialog
// ============================================================

interface ProfileDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  user: { nickname: string; email: string; phone: string };
  onSaved: () => void;
}

function ProfileDialog({ open, onOpenChange, user, onSaved }: ProfileDialogProps) {
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({ nickname: "", email: "", phone: "" });
  const [successMsg, setSuccessMsg] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  // 打开时同步当前用户信息
  const handleOpenChange = (nextOpen: boolean) => {
    if (nextOpen) {
      setForm({
        nickname: user.nickname || "",
        email: user.email || "",
        phone: user.phone || "",
      });
      setSuccessMsg("");
      setErrorMsg("");
    }
    onOpenChange(nextOpen);
  };

  const handleSave = async () => {
    setSaving(true);
    setSuccessMsg("");
    setErrorMsg("");
    try {
      await authApi.updateProfile({
        nickname: form.nickname || undefined,
        email: form.email || undefined,
        phone: form.phone || undefined,
      });
      setSuccessMsg("保存成功");
      onSaved();
      // 延迟关闭，让用户看到成功提示
      setTimeout(() => onOpenChange(false), 600);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "保存失败");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
        <DialogHeader className="shrink-0">
          <DialogTitle>编辑个人信息</DialogTitle>
          <DialogDescription>
            修改你的昵称、邮箱和手机号
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 overflow-y-auto min-h-0 px-1 -mx-1">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">昵称</Label>
            <Input
              placeholder="请输入昵称"
              value={form.nickname}
              onChange={e => setForm(prev => ({ ...prev, nickname: e.target.value }))}
              className="text-sm"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">邮箱地址</Label>
            <Input
              type="email"
              placeholder="请输入邮箱"
              value={form.email}
              onChange={e => setForm(prev => ({ ...prev, email: e.target.value }))}
              className="text-sm"
            />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">手机号码</Label>
            <Input
              type="tel"
              placeholder="请输入手机号"
              value={form.phone}
              onChange={e => setForm(prev => ({ ...prev, phone: e.target.value }))}
              className="text-sm"
            />
          </div>
          {/* 反馈消息 */}
          {successMsg && (
            <div className="flex items-center gap-1.5 text-xs text-green-500">
              <Check className="h-3.5 w-3.5" />
              {successMsg}
            </div>
          )}
          {errorMsg && (
            <p className="text-xs text-destructive">{errorMsg}</p>
          )}
        </div>

        <DialogFooter className="shrink-0">
          <DialogClose render={<Button variant="outline" size="sm" />}>
            取消
          </DialogClose>
          <Button size="sm" onClick={handleSave} disabled={saving}>
            {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
            保存
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================================
// 个人设置页面
// ============================================================

export default function ProfileSettingsPage() {
  const user = useAuthStore((s) => s.user);
  const fetchUserInfo = useAuthStore((s) => s.fetchUserInfo);

  // 个人信息 Dialog
  const [profileDialogOpen, setProfileDialogOpen] = useState(false);

  // 修改密码 — 内联表单
  const [pwForm, setPwForm] = useState({ oldPassword: "", newPassword: "", confirmPassword: "" });
  const [pwSaving, setPwSaving] = useState(false);
  const [pwSuccess, setPwSuccess] = useState("");
  const [pwError, setPwError] = useState("");
  const [showOldPw, setShowOldPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);

  // 用户名首字母（用于头像占位）
  const avatarInitial = (user?.nickname || user?.username || "U")
    .charAt(0)
    .toUpperCase();

  const handleChangePassword = async () => {
    setPwSuccess("");
    setPwError("");

    if (!pwForm.oldPassword.trim()) {
      setPwError("请输入旧密码");
      return;
    }
    if (pwForm.newPassword.length < 6) {
      setPwError("新密码长度须为 6 个字符以上");
      return;
    }
    if (pwForm.newPassword !== pwForm.confirmPassword) {
      setPwError("两次输入的新密码不一致");
      return;
    }

    setPwSaving(true);
    try {
      await authApi.changePassword({
        oldPassword: pwForm.oldPassword,
        newPassword: pwForm.newPassword,
      });
      setPwSuccess("密码修改成功");
      setPwForm({ oldPassword: "", newPassword: "", confirmPassword: "" });
    } catch (err: unknown) {
      setPwError(err instanceof Error ? err.message : "修改密码失败");
    } finally {
      setPwSaving(false);
    }
  };

  return (
    <motion.div
      className="max-w-[1200px]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* 页面标题 */}
      <motion.div variants={itemVariants} className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">个人设置</h1>
        <p className="text-muted-foreground mt-1">
          管理你的个人信息和账户安全
        </p>
      </motion.div>

      {/* ========== 个人信息卡片 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3 px-1">
          个人信息
        </h3>
        <div
          className={cn(
            "rounded-xl border border-border/30 overflow-hidden",
            "bg-card/50 backdrop-blur-sm"
          )}
        >
          {/* 头部：头像 + 基本信息 */}
          <div className="flex items-center gap-4 px-6 py-5">
            <div className="h-16 w-16 rounded-2xl bg-linear-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center text-white text-xl font-bold shadow-lg shrink-0">
              {avatarInitial}
            </div>
            <div className="flex-1 min-w-0">
              <h2 className="text-lg font-semibold">
                {user?.nickname || user?.username || "用户"}
              </h2>
              <p className="text-sm text-muted-foreground">
                @{user?.username || "unknown"}
              </p>
            </div>
            <button
              onClick={() => setProfileDialogOpen(true)}
              className={cn(
                "px-4 py-2 rounded-xl text-sm font-medium",
                "border border-border/30 bg-white/5",
                "text-muted-foreground hover:text-foreground hover:bg-white/10",
                "transition-all duration-200"
              )}
            >
              编辑资料
            </button>
          </div>

          {/* 详细信息条目 */}
          <div className="border-t border-border/20">
            {/* 昵称 */}
            <div className="flex items-center gap-3 px-6 py-3.5 border-b border-border/10">
              <div className="h-8 w-8 rounded-lg bg-blue-500/10 flex items-center justify-center shrink-0">
                <User className="h-4 w-4 text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-muted-foreground">昵称</p>
                <p className="text-sm font-medium">{user?.nickname || "未设置"}</p>
              </div>
            </div>

            {/* 邮箱 */}
            <div className="flex items-center gap-3 px-6 py-3.5 border-b border-border/10">
              <div className="h-8 w-8 rounded-lg bg-green-500/10 flex items-center justify-center shrink-0">
                <Mail className="h-4 w-4 text-green-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-muted-foreground">邮箱地址</p>
                <p className="text-sm font-medium">{user?.email || "未设置"}</p>
              </div>
            </div>

            {/* 手机号 */}
            <div className="flex items-center gap-3 px-6 py-3.5">
              <div className="h-8 w-8 rounded-lg bg-amber-500/10 flex items-center justify-center shrink-0">
                <Phone className="h-4 w-4 text-amber-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-muted-foreground">手机号码</p>
                <p className="text-sm font-medium">{user?.phone || "未设置"}</p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ========== 安全设置卡片 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3 px-1">
          安全设置
        </h3>
        <div
          className={cn(
            "rounded-xl border border-border/30 overflow-hidden",
            "bg-card/50 backdrop-blur-sm"
          )}
        >
          <div className="px-6 py-5">
            <div className="flex items-center gap-3 mb-5">
              <div className="h-9 w-9 rounded-lg bg-amber-500/10 flex items-center justify-center shrink-0">
                <Key className="h-4.5 w-4.5 text-amber-400" />
              </div>
              <div>
                <h4 className="text-sm font-medium">修改密码</h4>
                <p className="text-xs text-muted-foreground">
                  定期修改密码可以保护你的账户安全
                </p>
              </div>
            </div>

            <div className="space-y-3 max-w-md">
              {/* 旧密码 */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  旧密码 <span className="text-destructive">*</span>
                </Label>
                <div className="relative">
                  <Input
                    type={showOldPw ? "text" : "password"}
                    placeholder="请输入当前密码"
                    value={pwForm.oldPassword}
                    onChange={e => setPwForm(prev => ({ ...prev, oldPassword: e.target.value }))}
                    className="text-sm pr-9"
                  />
                  <button
                    type="button"
                    onClick={() => setShowOldPw(!showOldPw)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showOldPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              {/* 新密码 */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  新密码 <span className="text-destructive">*</span>
                </Label>
                <div className="relative">
                  <Input
                    type={showNewPw ? "text" : "password"}
                    placeholder="至少 6 个字符"
                    value={pwForm.newPassword}
                    onChange={e => setPwForm(prev => ({ ...prev, newPassword: e.target.value }))}
                    className="text-sm pr-9"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPw(!showNewPw)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                  >
                    {showNewPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              {/* 确认新密码 */}
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  确认新密码 <span className="text-destructive">*</span>
                </Label>
                <Input
                  type="password"
                  placeholder="再次输入新密码"
                  value={pwForm.confirmPassword}
                  onChange={e => setPwForm(prev => ({ ...prev, confirmPassword: e.target.value }))}
                  className="text-sm"
                />
              </div>

              {/* 反馈消息 */}
              {pwSuccess && (
                <div className="flex items-center gap-1.5 text-xs text-green-500 pt-1">
                  <Check className="h-3.5 w-3.5" />
                  {pwSuccess}
                </div>
              )}
              {pwError && (
                <p className="text-xs text-destructive pt-1">{pwError}</p>
              )}

              {/* 提交按钮 */}
              <div className="pt-2">
                <Button
                  size="sm"
                  onClick={handleChangePassword}
                  disabled={pwSaving || !pwForm.oldPassword || !pwForm.newPassword || !pwForm.confirmPassword}
                >
                  {pwSaving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
                  修改密码
                </Button>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ========== 账户信息卡片 ========== */}
      <motion.div variants={itemVariants} className="mb-8">
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider mb-3 px-1">
          账户信息
        </h3>
        <div
          className={cn(
            "rounded-xl border border-border/30 overflow-hidden",
            "bg-card/50 backdrop-blur-sm"
          )}
        >
          <div className="divide-y divide-border/10">
            <div className="flex items-center justify-between px-6 py-3.5">
              <span className="text-xs text-muted-foreground">用户名</span>
              <span className="text-sm font-mono">{user?.username || "—"}</span>
            </div>
            <div className="flex items-center justify-between px-6 py-3.5">
              <span className="text-xs text-muted-foreground">用户 ID</span>
              <span className="text-sm font-mono">{user?.id || "—"}</span>
            </div>
            <div className="flex items-center justify-between px-6 py-3.5">
              <span className="text-xs text-muted-foreground">角色</span>
              <div className="flex items-center gap-1.5">
                {user?.roles?.map((role) => (
                  <span
                    key={role}
                    className="px-2 py-0.5 rounded-full bg-primary/10 text-[10px] text-primary font-medium"
                  >
                    {role}
                  </span>
                )) || <span className="text-sm text-muted-foreground">—</span>}
              </div>
            </div>
            <div className="flex items-center justify-between px-6 py-3.5">
              <span className="text-xs text-muted-foreground">注册时间</span>
              <span className="text-sm font-mono">
                {user?.createTime
                  ? new Date(user.createTime).toLocaleDateString("zh-CN", {
                      year: "numeric",
                      month: "2-digit",
                      day: "2-digit",
                    })
                  : "—"}
              </span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* ========== 个人信息编辑 Dialog ========== */}
      <ProfileDialog
        open={profileDialogOpen}
        onOpenChange={setProfileDialogOpen}
        user={{
          nickname: user?.nickname || "",
          email: user?.email || "",
          phone: user?.phone || "",
        }}
        onSaved={() => fetchUserInfo()}
      />
    </motion.div>
  );
}
