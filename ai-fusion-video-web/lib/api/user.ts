import { http } from "./client";
import type { PageResult, UserRespVO } from "./types";

export interface UserPageQuery {
  pageNo?: number;
  pageSize?: number;
  username?: string;
  nickname?: string;
  status?: number;
}

export const userApi = {
  page: (params: UserPageQuery) =>
    http.get<never, PageResult<UserRespVO>>("/api/system/user/page", { params }),
};