ALTER TABLE afv_agent_conversation
    ADD agent_state_last_active_at DATETIME(3) NULL,
    ADD agent_state_expired_at DATETIME(3) NULL,
    ADD KEY idx_agent_conversation_state_retention (
        agent_state_expired_at,
        agent_state_last_active_at,
        id
    );

UPDATE afv_agent_conversation conversation
LEFT JOIN (
    SELECT
        conversation_id,
        MAX(COALESCE(finished_at, update_time)) AS last_active_at
    FROM afv_agent_run
    GROUP BY conversation_id
) latest_run ON latest_run.conversation_id = conversation.conversation_id
SET conversation.agent_state_last_active_at = COALESCE(
    latest_run.last_active_at,
    conversation.last_message_time,
    conversation.update_time,
    conversation.create_time
);

CREATE TABLE afv_agent_state_cleanup_policy (
    id BIGINT NOT NULL,
    cleanup_interval_days INT NOT NULL DEFAULT 1,
    retention_days INT NOT NULL DEFAULT 30,
    next_cleanup_at DATETIME(3) NOT NULL,
    cleanup_lease_owner VARCHAR(160) CHARACTER SET ascii COLLATE ascii_bin NULL,
    cleanup_lease_until DATETIME(3) NULL,
    last_cleanup_at DATETIME(3) NULL,
    create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    update_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
        ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    CONSTRAINT chk_agent_state_cleanup_interval
        CHECK (cleanup_interval_days BETWEEN 1 AND 365),
    CONSTRAINT chk_agent_state_retention_days
        CHECK (retention_days BETWEEN 1 AND 3650)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

INSERT INTO afv_agent_state_cleanup_policy (
    id,
    cleanup_interval_days,
    retention_days,
    next_cleanup_at
) VALUES (1, 1, 30, UTC_TIMESTAMP(3));
