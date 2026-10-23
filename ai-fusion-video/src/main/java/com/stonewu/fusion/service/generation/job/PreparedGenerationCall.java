package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.service.generation.transport.GenerationProviderFailure;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * 素材、请求体和客户端在提交意图之前准备；持久化层只保存 intent。
 * 连接、凭据及 SDK 请求对象仅保留在本次领取的内存中，不能进入持久化快照。
 */
public final class PreparedGenerationCall implements AutoCloseable {
    private final GenerationPreparedRequest intent;
    private final Supplier<GenerationProviderReceipt> operation;
    private final Runnable cleanup;
    private boolean submitted;
    private boolean closed;

    public PreparedGenerationCall(GenerationPreparedRequest intent, Supplier<GenerationProviderReceipt> operation) {
        this(intent, operation, () -> {
        });
    }

    public PreparedGenerationCall(GenerationPreparedRequest intent, Supplier<GenerationProviderReceipt> operation,
                                  Runnable cleanup) {
        this.intent = Objects.requireNonNull(intent);
        this.operation = Objects.requireNonNull(operation);
        this.cleanup = Objects.requireNonNull(cleanup);
    }

    public GenerationPreparedRequest intent() {
        return intent;
    }

    /**
     * 工作者已保存意图且重新校验身份后，至多调用一次。异常也不能在同一意图上重发。
     */
    public synchronized GenerationProviderReceipt submit() {
        if (closed || submitted) {
            throw new IllegalStateException("生成提交已经执行或关闭，不能再次调用");
        }
        submitted = true;
        try {
            return Objects.requireNonNull(operation.get(), "供应商必须返回提交回执");
        } catch (RuntimeException failure) {
            var response = GenerationProviderFailure.find(failure);
            if (response == null) {
                throw failure;
            }
            return new GenerationProviderReceipt(intent.providerTaskId(),
                    RemoteGenerationResult.failed(response.getMessage()));
        }
    }

    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            cleanup.run();
        }
    }
}
