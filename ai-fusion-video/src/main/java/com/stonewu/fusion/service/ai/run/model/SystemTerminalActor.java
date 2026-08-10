package com.stonewu.fusion.service.ai.run.model;

/** The complete allow-list of system actors permitted to request terminal CAS. */
public enum SystemTerminalActor {
    CANCELLATION_COORDINATOR,
    CONFIRMATION_EXPIRER,
    OWNER_RECONCILER
}
