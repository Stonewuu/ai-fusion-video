export interface RefreshSession {
  userId: number;
  accessToken: string;
  refreshToken: string;
}

export class RefreshRejectedError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

interface RefreshDependencies {
  read: () => RefreshSession | null;
  exchange: (session: RefreshSession) => Promise<RefreshSession>;
  publish: (previous: RefreshSession, next: RefreshSession) => Promise<void>;
  waitForRotation: (previous: RefreshSession) => Promise<RefreshSession>;
}

/** HTTP 与 SSE 共用一次刷新；失效请求不能覆盖退出或切换后的账户。 */
export function createSessionRefresher(dependencies: RefreshDependencies) {
  const pending = new Map<string, Promise<RefreshSession>>();

  return function refresh(failedToken: string | null, userId: number | null): Promise<RefreshSession> {
    const current = dependencies.read();
    if (!current || current.userId !== userId) {
      return Promise.reject(new Error("登录会话已变化，请重新登录"));
    }
    if (current.accessToken !== failedToken) {
      return Promise.resolve(current);
    }
    const existing = pending.get(current.refreshToken);
    if (existing) {
      return existing;
    }

    const operation = (async () => {
      let next: RefreshSession;
      try {
        next = await dependencies.exchange(current);
      } catch (error) {
        if (!(error instanceof RefreshRejectedError) || error.status !== 401) {
          throw error;
        }
        // 其他标签页可能已经消费同一凭据，只等待它发布真实的新会话，不再次提交旧凭据。
        return dependencies.waitForRotation(current);
      }
      const latest = dependencies.read();
      if (!latest || latest.userId !== current.userId || next.userId !== current.userId) {
        throw new Error("登录会话已变化，已放弃过期的刷新结果");
      }
      if (latest.refreshToken !== current.refreshToken) {
        return latest;
      }
      await dependencies.publish(current, next);
      return next;
    })().finally(() => {
      pending.delete(current.refreshToken);
    });
    pending.set(current.refreshToken, operation);
    return operation;
  };
}
