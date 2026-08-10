package com.stonewu.fusion.service.ai.agentscope.kernel;

import io.agentscope.core.model.ChatModelBase;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

public interface OwnedChatModel extends AutoCloseable {

    static OwnedChatModel owned(ChatModelBase model) {
        return new DefaultOwnedChatModel(model);
    }

    ChatModelBase model();

    @Override
    void close();
}

final class DefaultOwnedChatModel implements OwnedChatModel {
    private final ChatModelBase model;
    private final AtomicBoolean closed = new AtomicBoolean();

    DefaultOwnedChatModel(ChatModelBase model) {
        this.model = Objects.requireNonNull(model, "model must not be null");
    }

    @Override
    public ChatModelBase model() {
        return model;
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true) || !(model instanceof AutoCloseable closeable)) {
            return;
        }
        try {
            closeable.close();
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Exception failure) {
            throw new IllegalStateException("Failed to close AgentScope model", failure);
        }
    }
}
