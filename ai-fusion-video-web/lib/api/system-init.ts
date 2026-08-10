import { http } from "./client";
import type { LoginRespVO } from "./types";

// 系统初始化 API

// 初始化状态响应类型
export interface InitStatusResp {
  initialized: boolean;
  allowRegister: boolean;
}

// 初始化管理员请求类型
export interface SetupReqVO {
  username: string;
  password: string;
  nickname?: string;
}

/**
 * 获取系统初始化状态
 */
export function getInitStatus(): Promise<InitStatusResp> {
  return http.get<never, InitStatusResp>("/api/system/init/status");
}

/**
 * 初始化管理员账号（首次启动时）
 */
export function setupAdmin(data: SetupReqVO): Promise<LoginRespVO> {
  return http.post<never, LoginRespVO>("/api/system/init/setup", data);
}
