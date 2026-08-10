package com.stonewu.fusion.service.ai.agentscope.state;

public record StateStoreSlot(String userId, String sessionId) {

    private static final String ALL_SESSIONS = "<all-sessions>";
    private static final String STORE_LIFECYCLE = "<store-lifecycle>";

    public StateStoreSlot {
        userId = requireText(userId, "userId");
        sessionId = requireText(sessionId, "sessionId");
    }

    static StateStoreSlot allSessions(String userId) {
        return new StateStoreSlot(userId, ALL_SESSIONS);
    }

    static StateStoreSlot storeLifecycle() {
        return new StateStoreSlot(STORE_LIFECYCLE, STORE_LIFECYCLE);
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value.trim();
    }
}
