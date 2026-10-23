package com.stonewu.fusion.service.generation.job;

/**
 * 只在已领取的工作者调用栈上传递写入凭据，不是任务真相或调度队列。
 */
public final class GenerationLeaseScope implements AutoCloseable {
    private static final ThreadLocal<RemoteGenerationJob.Claim> CURRENT = new ThreadLocal<>();
    private final RemoteGenerationJob.Claim previous;
    private final RemoteGenerationJob.Claim claim;

    private GenerationLeaseScope(RemoteGenerationJob.Claim claim) {
        this.previous = CURRENT.get();
        this.claim = claim;
        CURRENT.set(claim);
    }

    public static GenerationLeaseScope enter(RemoteGenerationJob.Claim claim) {
        if (claim == null) {
            throw GenerationJobException.fenced();
        }
        return new GenerationLeaseScope(claim);
    }

    public static RemoteGenerationJob.Claim require() {
        var claim = CURRENT.get();
        if (claim == null) {
            throw GenerationJobException.fenced();
        }
        return claim;
    }

    @Override
    public void close() {
        if (CURRENT.get() != claim) {
            throw new IllegalStateException("生成租约调用栈顺序错误");
        }
        if (previous == null) {
            CURRENT.remove();
        } else {
            CURRENT.set(previous);
        }
    }
}
