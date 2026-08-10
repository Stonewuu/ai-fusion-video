import { create } from "zustand";
import { persist } from "zustand/middleware";
import * as authApi from "@/lib/api/auth";
import type { UserRespVO } from "@/lib/api/types";

// 认证状态类型
interface AuthState {
  // 状态
  token: string | null;
  refreshToken: string | null;
  user: UserRespVO | null;

  // 计算属性
  isAuthenticated: () => boolean;

  // Actions
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  fetchUserInfo: () => Promise<void>;
  setTokens: (accessToken: string, refreshToken: string) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      refreshToken: null,
      user: null,

      isAuthenticated: () => !!get().token,

      // 登录
      login: async (username: string, password: string) => {
        const resp = await authApi.login({ username, password });
        set({
          token: resp.accessToken,
          refreshToken: resp.refreshToken,
          user: {
            id: resp.userId,
            username: resp.username,
            nickname: resp.nickname,
            avatar: null,
            email: null,
            phone: null,
            status: 0,
            createTime: "",
            roles: [],
          },
        });
        // 登录后获取完整用户信息
        try {
          await get().fetchUserInfo();
        } catch {
          // 即使获取用户信息失败，登录仍然成功
        }
      },

      // 登出
      logout: async () => {
        try {
          await authApi.logout();
        } catch {
          // 即使后端登出失败，前端也清除状态
        }
        set({ token: null, refreshToken: null, user: null });
      },

      // 获取用户信息
      fetchUserInfo: async () => {
        const user = await authApi.getUserInfo();
        set({ user });
      },

      // 更新令牌对（刷新令牌后调用）
      setTokens: (accessToken: string, refreshToken: string) => {
        set({ token: accessToken, refreshToken });
      },

      // 清除认证状态（无需调后端）
      clearAuth: () => {
        set({ token: null, refreshToken: null, user: null });
      },
    }),
    {
      name: "auth-storage",
      // 只持久化 token、refreshToken 和用户基本信息
      partialize: (state) => ({
        token: state.token,
        refreshToken: state.refreshToken,
        user: state.user,
      }),
    }
  )
);
