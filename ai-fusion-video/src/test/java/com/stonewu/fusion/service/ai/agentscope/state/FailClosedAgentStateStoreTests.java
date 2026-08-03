package com.stonewu.fusion.service.ai.agentscope.state;

import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.State;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.catchThrowable;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class FailClosedAgentStateStoreTests {

    private final AgentStateStore delegate = mock(AgentStateStore.class);
    private final InMemoryStateStoreFailureGuard guard = new InMemoryStateStoreFailureGuard();
    private final FailClosedAgentStateStore store = new FailClosedAgentStateStore(delegate, guard);

    @Test
    void delegatesTheCompleteAgentStateStoreContract() {
        TestState state = new TestState("one");
        List<TestState> states = List.of(state, new TestState("two"));
        when(delegate.get("42", "session-a", "single", TestState.class)).thenReturn(Optional.of(state));
        when(delegate.getList("42", "session-a", "list", TestState.class)).thenReturn(states);
        when(delegate.exists("42", "session-a")).thenReturn(true);
        when(delegate.listSessionIds("42")).thenReturn(Set.of("session-a"));

        store.save("42", "session-a", "single", state);
        store.save("42", "session-a", "list", states);
        assertThat(store.get("42", "session-a", "single", TestState.class)).contains(state);
        assertThat(store.getList("42", "session-a", "list", TestState.class)).containsExactlyElementsOf(states);
        assertThat(store.exists("42", "session-a")).isTrue();
        store.delete("42", "session-a", "single");
        store.delete("42", "session-a");
        assertThat(store.listSessionIds("42")).containsExactly("session-a");
        store.close();

        verify(delegate).save("42", "session-a", "single", state);
        verify(delegate).save("42", "session-a", "list", states);
        verify(delegate).get("42", "session-a", "single", TestState.class);
        verify(delegate).getList("42", "session-a", "list", TestState.class);
        verify(delegate).exists("42", "session-a");
        verify(delegate).delete("42", "session-a", "single");
        verify(delegate).delete("42", "session-a");
        verify(delegate).listSessionIds("42");
        verify(delegate).close();
    }

    @Test
    void recordsAndRethrowsTheFirstFailureForTheExactSlot() {
        StateStoreSlot slot = new StateStoreSlot("42", "session-a");
        IllegalStateException loadCause = new IllegalStateException("load failed");
        IllegalArgumentException saveCause = new IllegalArgumentException("save failed later");
        when(delegate.get("42", "session-a", "state", TestState.class)).thenThrow(loadCause);
        doThrow(saveCause).when(delegate).save("42", "session-a", "state", new TestState("later"));

        Throwable first = catchThrowable(
                () -> store.get("42", "session-a", "state", TestState.class));
        Throwable repeated = catchThrowable(
                () -> store.save("42", "session-a", "state", new TestState("later")));

        assertThat(first).isInstanceOf(StateStoreFailure.class);
        assertThat(repeated).isSameAs(first);
        assertThat(guard.failure(slot)).containsSame((StateStoreFailure) first);
        assertThat(((StateStoreFailure) first).operation()).isEqualTo("get");
        assertThat(first.getCause()).isSameAs(loadCause);
        assertThat(first.getCause()).isNotSameAs(saveCause);
    }

    @Test
    void neverDoubleWrapsAStateStoreFailure() {
        StateStoreSlot slot = new StateStoreSlot("42", "session-a");
        StateStoreFailure existing = new StateStoreFailure(
                slot, "save", new IllegalStateException("redis down"));
        doThrow(existing).when(delegate).save("42", "session-a", "state", new TestState("one"));

        Throwable thrown = catchThrowable(
                () -> store.save("42", "session-a", "state", new TestState("one")));

        assertThat(thrown).isSameAs(existing);
        assertThat(guard.failure(slot)).containsSame(existing);
    }

    @Test
    void keepsDifferentSessionsIndependent() {
        when(delegate.exists("42", "session-a")).thenThrow(new IllegalStateException("a failed"));
        when(delegate.exists("42", "session-b")).thenReturn(true);

        Throwable failure = catchThrowable(() -> store.exists("42", "session-a"));

        assertThat(failure).isInstanceOf(StateStoreFailure.class);
        assertThat(store.exists("42", "session-b")).isTrue();
        assertThat(guard.failure(new StateStoreSlot("42", "session-b"))).isEmpty();
    }

    @Test
    void treatsFrameworkAnonymousBootstrapReadsAsAbsentWithoutPersistingAnAnonymousSlot() {
        assertThat(store.get(null, "agent-name", "agent_state", TestState.class)).isEmpty();
        assertThat(store.get(null, "agent-name", "toolkit_activeGroups", TestState.class)).isEmpty();
        assertThat(store.getList(null, "agent-name", "memory_messages", TestState.class)).isEmpty();
        assertThatThrownBy(() -> store.exists(null, "agent-name"))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> store.get(null, "agent-name", "other", TestState.class))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> store.get(" ", "agent-name", "agent_state", TestState.class))
                .isInstanceOf(IllegalArgumentException.class);

        verify(delegate, never()).get(null, "agent-name", "agent_state", TestState.class);
        verify(delegate, never()).get(null, "agent-name", "toolkit_activeGroups", TestState.class);
        verify(delegate, never()).getList(null, "agent-name", "memory_messages", TestState.class);
        verify(delegate, never()).exists(null, "agent-name");
    }

    private record TestState(String value) implements State {
    }
}
