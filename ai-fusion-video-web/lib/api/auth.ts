import { http } from "./client";
import type { LoginReqVO, LoginRespVO, RegisterReqVO, UserRespVO, ProfileUpdateReq, ChangePasswordReq } from "./types";

// 认证相关 API

/**
 * 登录
 */
export function login(data: LoginReqVO): Promise<LoginRespVO> {
  return http.post<never, LoginRespVO>("/api/auth/login", data);
}

/**
 * 注册
 */
export function register(data: RegisterReqVO): Promise<LoginRespVO> {
  return http.post<never, LoginRespVO>("/api/auth/register", data);
}

/**
 * 获取当前用户信息
 */
export function getUserInfo(): Promise<UserRespVO> {
  return http.get<never, UserRespVO>("/api/auth/user-info");
}

/**
 * 登出
 */
export function logout(): Promise<boolean> {
  return http.post<never, boolean>("/api/auth/logout");
}

/**
 * 使用 refresh_token 刷新令牌
 */
export function refreshToken(refreshToken: string): Promise<LoginRespVO> {
  return http.post<never, LoginRespVO>("/api/auth/refresh", { refreshToken });
}

/**
 * 更新个人资料
 */
export function updateProfile(data: ProfileUpdateReq): Promise<boolean> {
  return http.put<never, boolean>("/api/auth/profile", data);
}

/**
 * 修改密码
 */
export function changePassword(data: ChangePasswordReq): Promise<boolean> {
  return http.put<never, boolean>("/api/auth/change-password", data);
}
