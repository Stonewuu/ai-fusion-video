"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { AuthLayout } from "@/components/ui/auth-layout";
import { useAuthStore } from "@/lib/store/auth-store";
import { getInitStatus } from "@/lib/api/system-init";
import { getApiErrorMessage } from "@/lib/api/api-error";
import { register as registerApi } from "@/lib/api/auth";

export default function RegisterPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [nickname, setNickname] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [initReady, setInitReady] = useState(false);
  const [allowRegister, setAllowRegister] = useState(false);

  useEffect(() => {
    getInitStatus()
      .then((status) => {
        if (!status.initialized) {
          router.replace("/setup");
          return;
        }
        setAllowRegister(status.allowRegister);
        setInitReady(true);
      })
      .catch((error) => {
        setError(getApiErrorMessage(error));
        setInitReady(true);
      });
  }, [router]);

  const validate = (): string | null => {
    if (!username.trim()) return "请输入用户名";
    if (username.trim().length < 3) return "用户名至少 3 个字符";
    if (!password) return "请输入密码";
    if (password.length < 6) return "密码至少 6 位";
    if (password !== confirmPassword) return "两次输入的密码不一致";
    return null;
  };

  const isFormValid =
    username.trim().length >= 3 &&
    password.length >= 6 &&
    password === confirmPassword;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }

    setError("");
    setLoading(true);

    try {
      const resp = await registerApi({
        username: username.trim(),
        password,
        confirmPassword,
        nickname: nickname.trim() || undefined,
      });

      useAuthStore.setState({
        token: resp.accessToken,
        refreshToken: resp.refreshToken,
        user: {
          id: resp.userId,
          username: resp.username,
          nickname: resp.nickname,
          avatar: null,
          email: null,
          phone: null,
          status: 1,
          createTime: "",
          roles: ["user"],
        },
      });

      document.cookie = `auth-token=${resp.accessToken}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;
      setShowSuccess(true);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("注册失败，请稍后重试");
      }
    } finally {
      setLoading(false);
    }
  };

  if (!initReady) {
    return <div className="min-h-screen bg-black" />;
  }

  if (!allowRegister) {
    return (
      <AuthLayout>
        <div className="space-y-4">
          <h1 className="text-[2rem] font-bold leading-[1.1] tracking-tight text-white">
            当前未开放注册
          </h1>
          <p className="text-base text-white/50 font-light leading-relaxed">
            需要管理员先完成初始化，并在系统设置中开启允许注册。
          </p>
          {error ? <p className="text-sm text-red-400/90">{error}</p> : null}
          <Link
            href="/login"
            className="inline-flex items-center justify-center rounded-full bg-white px-5 py-3 text-sm font-medium text-black transition-colors hover:bg-white/90"
          >
            返回登录
          </Link>
        </div>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      showSuccess={showSuccess}
      successTitle="注册成功"
      successSubtitle="正在进入控制面板"
      onTransitionComplete={() => router.replace("/dashboard")}
    >
      <div className="space-y-2">
        <h1 className="text-[2rem] font-bold leading-[1.1] tracking-tight text-white">
          创建账号
        </h1>
        <p className="text-base text-white/50 font-light">使用用户名和密码创建账号</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="text"
          placeholder="用户名"
          value={username}
          onChange={(e) => {
            setUsername(e.target.value);
            setError("");
          }}
          className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
          required
          autoComplete="username"
          disabled={loading}
          minLength={3}
        />

        <input
          type="text"
          placeholder="昵称（选填）"
          value={nickname}
          onChange={(e) => setNickname(e.target.value)}
          className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
          autoComplete="nickname"
          disabled={loading}
        />

        <input
          type="password"
          placeholder="密码（至少6位）"
          value={password}
          onChange={(e) => {
            setPassword(e.target.value);
            setError("");
          }}
          className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
          required
          autoComplete="new-password"
          minLength={6}
          disabled={loading}
        />

        <input
          type="password"
          placeholder="确认密码"
          value={confirmPassword}
          onChange={(e) => {
            setConfirmPassword(e.target.value);
            setError("");
          }}
          className="w-full backdrop-blur-[1px] text-white border border-white/10 rounded-full py-3 px-5 focus:outline-none focus:border-white/30 bg-transparent placeholder:text-white/30 transition-colors"
          required
          autoComplete="new-password"
          minLength={6}
          disabled={loading}
        />

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <p className="text-red-400/90 text-sm py-1">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        <motion.button
          type="submit"
          disabled={loading || !isFormValid}
          className={cn(
            "w-full rounded-full font-medium py-3 transition-all duration-300",
            loading || !isFormValid
              ? "bg-[#111] text-white/50 border border-white/10 cursor-not-allowed"
              : "bg-white text-black hover:bg-white/90 cursor-pointer"
          )}
          whileHover={!loading && isFormValid ? { scale: 1.02 } : undefined}
          whileTap={!loading && isFormValid ? { scale: 0.98 } : undefined}
          transition={{ duration: 0.2 }}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              注册中...
            </span>
          ) : (
            "注册并开始使用"
          )}
        </motion.button>
      </form>

      <p className="text-sm text-white/45 pt-2">
        已有账号？{" "}
        <Link href="/login" className="text-white underline decoration-white/30 underline-offset-4 hover:decoration-white/80 transition-colors">
          去登录
        </Link>
      </p>

      <p className="text-xs text-white/30 pt-4">融光 · 公开注册</p>
    </AuthLayout>
  );
}
