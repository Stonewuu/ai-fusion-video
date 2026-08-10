"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { AuthLayout } from "@/components/ui/auth-layout";
import { http } from "@/lib/api/client";

function ForgotPasswordContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  const tokenParam = searchParams.get("token") || "";
  const typeParam = searchParams.get("type") || "";

  // 页面状态：
  // 'request': 初始输入用户名和选择找回方式
  // 'verify_loading': 邮件链接加载校验中
  // 'reset_form': 输入新密码（邮件链接验证通过后，或日志方式点击发送后直接转为此状态）
  // 'success_message': 邮件发送成功提示
  const [step, setStep] = useState<"request" | "verify_loading" | "reset_form" | "success_message">("request");

  // 表单状态
  const [usernameOrEmail, setUsernameOrEmail] = useState("");
  const [resetType, setResetType] = useState<"email" | "log">("email");
  const [token, setToken] = useState("");
  const [verifiedUsername, setVerifiedUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [successInfo, setSuccessInfo] = useState("");
  const [showSuccessAnim, setShowSuccessAnim] = useState(false);

  // 如果 URL 里有 token 并且是邮箱类型，直接去校验
  useEffect(() => {
    if (tokenParam && typeParam === "email") {
      setStep("verify_loading");
      setToken(tokenParam);
      setResetType("email");
      setError("");

      http
        .post<never, string>(
          `/api/auth/reset-password/verify?token=${encodeURIComponent(tokenParam)}&type=email`
        )
        .then((username) => {
          setVerifiedUsername(username);
          setStep("reset_form");
        })
        .catch((err: unknown) => {
          const errorMsg = (err as { message?: string })?.message;
          setError(errorMsg || "重置链接无效或已过期，请重新申请");
          setStep("request");
        });
    }
  }, [tokenParam, typeParam]);

  // 发起密码重置申请
  const handleRequestSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameOrEmail) return;

    setError("");
    setLoading(true);

    try {
      const msg = await http.post<never, string>("/api/auth/reset-password/request", {
        usernameOrEmail,
        type: resetType,
      });

      if (resetType === "email") {
        setSuccessInfo(msg || "重置链接已成功发送至您的邮箱！");
        setStep("success_message");
      } else {
        // 日志找回不需要等待邮件，在当前页面直接进入“输入验证码重置密码”阶段
        setSuccessInfo("验证码已发送至系统后台日志，请获取并输入：");
        setStep("reset_form");
      }
    } catch (err: unknown) {
      const errorMsg = (err as { message?: string })?.message;
      setError(errorMsg || "申请密码找回失败，请检查用户名是否正确");
    } finally {
      setLoading(false);
    }
  };

  // 提交重置密码
  const handleResetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const activeToken = resetType === "log" ? token : tokenParam;
    if (!activeToken || !newPassword || !confirmPassword) return;

    if (newPassword !== confirmPassword) {
      setError("两次输入的密码不一致");
      return;
    }

    setError("");
    setLoading(true);

    try {
      await http.post("/api/auth/reset-password/submit", {
        token: activeToken,
        type: resetType,
        newPassword,
        confirmPassword,
      });

      // 触发 AuthLayout 成功跳转动画
      setShowSuccessAnim(true);
    } catch (err: unknown) {
      const errorMsg = (err as { message?: string })?.message;
      setError(errorMsg || "重置密码失败，请确认验证码正确或重新申请");
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      showSuccess={showSuccessAnim}
      successTitle="密码重置成功"
      successSubtitle="正在跳转至登录页面"
      onTransitionComplete={() => router.replace("/login")}
    >
      {/* 标题控制 */}
      <div className="space-y-2">
        <h1 className="text-[2rem] font-bold leading-[1.1] tracking-tight text-white">
          {step === "request" && "找回密码"}
          {step === "verify_loading" && "验证凭证中"}
          {step === "reset_form" && "重置您的密码"}
          {step === "success_message" && "邮件已发送"}
        </h1>
        <p className="text-base text-white/50 font-light">
          {step === "request" && "选择安全的方式来恢复您的密码"}
          {step === "verify_loading" && "正在校验您的安全重置链接，请稍候"}
          {step === "reset_form" && (verifiedUsername ? `正在为账号 "${verifiedUsername}" 重置密码` : "请输入新密码以完成密码重置")}
          {step === "success_message" && "重置密码指引已发送"}
        </p>
      </div>

      {/* 错误提示 */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm py-2.5 px-4 rounded-2xl mb-2">
              {error}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 第一阶段：发起密码重置申请 */}
      {step === "request" && (
        <form onSubmit={handleRequestSubmit} className="space-y-4">
          <div className="relative">
            <input
              type="text"
              placeholder="请输入用户名或邮箱地址"
              value={usernameOrEmail}
              onChange={(e) => {
                setUsernameOrEmail(e.target.value);
                setError("");
              }}
              className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
              required
              disabled={loading}
            />
          </div>

          {/* 找回方式选择 */}
          <div className="grid grid-cols-2 gap-3 pt-1">
            <button
              type="button"
              onClick={() => setResetType("email")}
              className={cn(
                "py-3 px-4 rounded-full border text-sm font-medium transition-all duration-300",
                resetType === "email"
                  ? "bg-white text-black border-white"
                  : "bg-transparent text-white/60 border-white/10 hover:border-white/30"
              )}
            >
              邮箱链接找回
            </button>
            <button
              type="button"
              onClick={() => setResetType("log")}
              className={cn(
                "py-3 px-4 rounded-full border text-sm font-medium transition-all duration-300",
                resetType === "log"
                  ? "bg-white text-black border-white"
                  : "bg-transparent text-white/60 border-white/10 hover:border-white/30"
              )}
            >
              日志验证码找回
            </button>
          </div>

          <motion.button
            type="submit"
            disabled={loading || !usernameOrEmail}
            className={cn(
              "w-full rounded-full font-medium py-3 mt-2 transition-all duration-300",
              loading || !usernameOrEmail
                ? "bg-[#111] text-white/50 border border-white/10 cursor-not-allowed"
                : "bg-white text-black hover:bg-white/90 cursor-pointer"
            )}
            whileHover={!loading && usernameOrEmail ? { scale: 1.02 } : undefined}
            whileTap={!loading && usernameOrEmail ? { scale: 0.98 } : undefined}
          >
            {loading ? "正在申请..." : "发起找回"}
          </motion.button>
        </form>
      )}

      {/* 校验加载中状态 */}
      {step === "verify_loading" && (
        <div className="flex justify-center items-center py-8">
          <svg className="animate-spin h-8 w-8 text-white" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        </div>
      )}

      {/* 第二阶段：输入验证码/凭证及密码重置表单 */}
      {step === "reset_form" && (
        <form onSubmit={handleResetSubmit} className="space-y-4">
          {successInfo && (
            <p className="text-xs text-emerald-400/90 leading-relaxed bg-emerald-500/5 border border-emerald-500/10 p-3 rounded-2xl">
              {successInfo}
            </p>
          )}

          {/* 如果是日志验证方式，需要手动输入验证码（Token） */}
          {resetType === "log" && (
            <div className="relative">
              <input
                type="text"
                placeholder="请输入日志中显示的验证码"
                value={token}
                onChange={(e) => {
                  setToken(e.target.value);
                  setError("");
                }}
                className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
                required
                disabled={loading}
              />
            </div>
          )}

          <div className="relative">
            <input
              type="password"
              placeholder="请输入新密码"
              value={newPassword}
              onChange={(e) => {
                setNewPassword(e.target.value);
                setError("");
              }}
              className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
              required
              minLength={6}
              disabled={loading}
            />
          </div>

          <div className="relative">
            <input
              type="password"
              placeholder="请再次确认新密码"
              value={confirmPassword}
              onChange={(e) => {
                setConfirmPassword(e.target.value);
                setError("");
              }}
              className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
              required
              minLength={6}
              disabled={loading}
            />
          </div>

          <motion.button
            type="submit"
            disabled={loading || (resetType === "log" ? !token : false) || !newPassword || !confirmPassword}
            className={cn(
              "w-full rounded-full font-medium py-3 mt-2 transition-all duration-300",
              loading || (resetType === "log" ? !token : false) || !newPassword || !confirmPassword
                ? "bg-[#111] text-white/50 border border-white/10 cursor-not-allowed"
                : "bg-white text-black hover:bg-white/90 cursor-pointer"
            )}
            whileHover={!loading ? { scale: 1.02 } : undefined}
            whileTap={!loading ? { scale: 0.98 } : undefined}
          >
            {loading ? "正在重置..." : "重置密码"}
          </motion.button>
        </form>
      )}

      {/* 邮件发送成功提示页 */}
      {step === "success_message" && (
        <div className="space-y-4">
          <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm py-4 px-5 rounded-2xl leading-relaxed">
            {successInfo}
          </div>
          <p className="text-xs text-white/40">
            如果您在几分钟内没有收到邮件，请检查您的垃圾邮件文件夹，或确认该账户是否绑定了正确的邮箱地址。
          </p>
          <button
            onClick={() => setStep("request")}
            className="w-full rounded-full border border-white/15 text-white py-3 hover:bg-white/5 transition-all text-sm font-medium"
          >
            重新申请
          </button>
        </div>
      )}

      {/* 页面底部返回登录入口 */}
      <div className="pt-2 text-sm text-white/45">
        记起密码了？{" "}
        <Link href="/login" className="text-white underline decoration-white/30 underline-offset-4 hover:decoration-white/80 transition-colors">
          立即登录
        </Link>
      </div>

      <p className="text-xs text-white/30 pt-8">融光 · AI视频创作平台</p>
    </AuthLayout>
  );
}

export default function ForgotPasswordPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-black" />}>
      <ForgotPasswordContent />
    </Suspense>
  );
}
