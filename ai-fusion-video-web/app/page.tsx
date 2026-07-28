"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { getInitStatus } from "@/lib/api/system-init";
import { getApiErrorMessage } from "@/lib/api/api-error";

// 根页面：仅检测初始化状态并重定向
// 认证保护由 middleware 处理，已登录用户不会到达此页面
export default function Home() {
  const router = useRouter();
  const [error, setError] = useState("");

  useEffect(() => {
    getInitStatus()
      .then((status) => {
        if (!status.initialized) {
          router.replace("/setup");
        } else {
          router.replace("/login");
        }
      })
      .catch((error) => {
        // 后端不可用时直接跳登录页
        setError(getApiErrorMessage(error));
        setTimeout(() => router.replace("/login"), 2000);
      });
  }, [router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-black">
      {error ? (
        <p className="text-white/50 text-sm">{error}</p>
      ) : (
        <div className="animate-spin h-6 w-6 border-2 border-white/20 border-t-white/80 rounded-full" />
      )}
    </div>
  );
}
