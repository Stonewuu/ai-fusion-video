package com.stonewu.fusion.service.ai.agentscope.state;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class StateStoreFailureGuardTests {

    private final InMemoryStateStoreFailureGuard guard = new InMemoryStateStoreFailureGuard();

    @Test
    void firstFailureWinsAndTheSameInstanceRemainsUntilExplicitClear() {
        StateStoreSlot slot = new StateStoreSlot(" user-42 ", " session-7 ");
        IllegalStateException originalCause = new IllegalStateException("redis down");
        IllegalArgumentException laterCause = new IllegalArgumentException("later failure");

        StateStoreFailure first = guard.record(slot, "load", originalCause);
        StateStoreFailure repeated = guard.record(slot, "save", laterCause);

        assertThat(repeated).isSameAs(first);
        assertThat(first.slot()).isEqualTo(new StateStoreSlot("user-42", "session-7"));
        assertThat(first.operation()).isEqualTo("load");
        assertThat(first.getCause()).isSameAs(originalCause);
        assertThat(guard.failure(slot)).containsSame(first);
        assertThatThrownBy(() -> guard.throwIfFailed(slot)).isSameAs(first);
        assertThatThrownBy(() -> guard.throwIfFailed(slot)).isSameAs(first);
        assertThat(guard.failure(slot)).containsSame(first);

        guard.clear(slot);

        assertThat(guard.failure(slot)).isEmpty();
        guard.throwIfFailed(slot);
        StateStoreFailure afterClear = guard.record(slot, "save", laterCause);
        assertThat(afterClear).isNotSameAs(first);
        assertThat(afterClear.getCause()).isSameAs(laterCause);
    }

    @Test
    void failuresAreIsolatedByExactUserAndSessionSlot() {
        StateStoreSlot firstSlot = new StateStoreSlot("42", "session-a");
        StateStoreSlot secondSlot = new StateStoreSlot("42", "session-b");
        StateStoreFailure first = guard.record(firstSlot, "get", new IllegalStateException("first"));
        StateStoreFailure second = guard.record(secondSlot, "get", new IllegalStateException("second"));

        assertThat(first).isNotSameAs(second);
        assertThat(guard.failure(firstSlot)).containsSame(first);
        assertThat(guard.failure(secondSlot)).containsSame(second);

        guard.clear(firstSlot);

        assertThat(guard.failure(firstSlot)).isEmpty();
        assertThat(guard.failure(secondSlot)).containsSame(second);
    }

    @Test
    void doesNotWrapAnAlreadyRecordedFailureForTheSameSlot() {
        StateStoreSlot slot = new StateStoreSlot("42", "session-a");
        StateStoreFailure existing = new StateStoreFailure(
                slot, "save", new IllegalStateException("redis down"));

        StateStoreFailure stored = guard.record(slot, "save", existing);

        assertThat(stored).isSameAs(existing);
        assertThat(stored.getCause()).isNotInstanceOf(StateStoreFailure.class);
    }

    @Test
    void validatesFailureIdentity() {
        StateStoreSlot slot = new StateStoreSlot("42", "session-a");

        assertThatThrownBy(() -> new StateStoreSlot(" ", "session-a"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("userId");
        assertThatThrownBy(() -> guard.record(slot, " ", new IllegalStateException()))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("operation");
        StateStoreFailure existing = new StateStoreFailure(
                slot, "save", new IllegalStateException("existing"));
        assertThatThrownBy(() -> guard.record(slot, " ", existing))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("operation");
        assertThatThrownBy(() -> guard.record(slot, "save", null))
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("cause");
    }
}
