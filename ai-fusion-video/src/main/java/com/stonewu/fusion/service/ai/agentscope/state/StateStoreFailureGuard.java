package com.stonewu.fusion.service.ai.agentscope.state;

import java.util.Optional;

public interface StateStoreFailureGuard {

    void clear(StateStoreSlot slot);

    StateStoreFailure record(StateStoreSlot slot, String operation, Throwable cause);

    Optional<StateStoreFailure> failure(StateStoreSlot slot);

    void throwIfFailed(StateStoreSlot slot) throws StateStoreFailure;
}
