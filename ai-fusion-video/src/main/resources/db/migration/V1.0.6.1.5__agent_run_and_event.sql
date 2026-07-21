-- AgentScope V2 durable runtime requires enforced CHECK constraints.
-- Fail before any DDL on unsupported MySQL or MariaDB variants.
SET @afv_mysql_major = CAST(SUBSTRING_INDEX(VERSION(), '.', 1) AS UNSIGNED);
SET @afv_mysql_minor = CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(VERSION(), '.', 2), '.', -1) AS UNSIGNED);
SET @afv_mysql_patch = CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(VERSION(), '-', 1), '.', -1) AS UNSIGNED);
SET @afv_mysql_supported = VERSION() NOT LIKE '%MariaDB%'
    AND (@afv_mysql_major > 8
        OR (@afv_mysql_major = 8 AND @afv_mysql_minor > 0)
        OR (@afv_mysql_major = 8 AND @afv_mysql_minor = 0 AND @afv_mysql_patch >= 16));
SET @afv_mysql_guard_sql = IF(
    @afv_mysql_supported,
    'SELECT 1',
    'SELECT * FROM __AGENTSCOPE_REQUIRES_MYSQL_8_0_16_OR_NEWER__');
PREPARE afv_mysql_guard FROM @afv_mysql_guard_sql;
EXECUTE afv_mysql_guard;
DEALLOCATE PREPARE afv_mysql_guard;

-- Rewrite the complete message history deterministically before adding the
-- unique order constraint. Deleted messages intentionally retain an order.
CREATE TEMPORARY TABLE afv_order_rewrite (
    id BIGINT PRIMARY KEY,
    new_order BIGINT NOT NULL
);

INSERT INTO afv_order_rewrite(id, new_order)
SELECT
    id,
    ROW_NUMBER() OVER (
        PARTITION BY conversation_id
        ORDER BY message_order, id)
FROM afv_agent_message;

UPDATE afv_agent_message message
JOIN afv_order_rewrite rewrite ON rewrite.id = message.id
SET message.message_order = rewrite.new_order;

DROP TEMPORARY TABLE afv_order_rewrite;

ALTER TABLE afv_agent_message
    MODIFY message_order BIGINT NOT NULL,
    ADD run_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    ADD projection_key VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    DROP INDEX idx_msg_conv_order,
    ADD UNIQUE KEY uk_agent_message_conv_order(conversation_id, message_order),
    ADD UNIQUE KEY uk_agent_message_projection_key(projection_key),
    ADD KEY idx_agent_message_conv_run_order(conversation_id, run_id, message_order);

ALTER TABLE afv_agent_conversation
    ADD next_message_order BIGINT NOT NULL DEFAULT 1;

UPDATE afv_agent_conversation conversation
LEFT JOIN (
    SELECT
        conversation_id,
        COALESCE(MAX(message_order), 0) + 1 AS next_order,
        SUM(deleted = 0) AS visible_count
    FROM afv_agent_message
    GROUP BY conversation_id
) messages ON messages.conversation_id = conversation.conversation_id
SET
    conversation.next_message_order = COALESCE(messages.next_order, 1),
    conversation.message_count = COALESCE(messages.visible_count, 0);

CREATE TABLE afv_agent_run (
    id BIGINT NOT NULL AUTO_INCREMENT,
    run_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    conversation_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    user_id BIGINT NOT NULL,
    project_id BIGINT NULL,
    agent_type VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    parent_run_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    parent_tool_call_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    agent_name VARCHAR(128) NULL,
    kernel_fingerprint VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    agent_definition_snapshot_json MEDIUMTEXT NOT NULL,
    agent_state_session_id VARCHAR(255) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    status VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    owner_instance_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    owner_epoch BIGINT NOT NULL DEFAULT 0,
    lease_until DATETIME(3) NULL,
    next_sequence BIGINT NOT NULL DEFAULT 1,
    terminal_sequence BIGINT NULL,
    terminal_output_type VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NULL,
    cancel_requested_at DATETIME(3) NULL,
    cancel_broadcast_at DATETIME(3) NULL,
    cancel_acknowledged_at DATETIME(3) NULL,
    cancel_next_attempt_at DATETIME(3) NULL,
    waiting_reply_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    waiting_tool_call_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    waiting_tool_name VARCHAR(128) NULL,
    wait_expires_at DATETIME(3) NULL,
    paused_through_sequence BIGINT NULL,
    deadline_at DATETIME(3) NOT NULL,
    started_at DATETIME(3) NOT NULL,
    heartbeat_at DATETIME(3) NULL,
    finished_at DATETIME(3) NULL,
    error_code VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL,
    error_message TEXT NULL,
    usage_settled TINYINT NOT NULL DEFAULT 0,
    usage_settled_at DATETIME(3) NULL,
    projected_through_sequence BIGINT NOT NULL DEFAULT 0,
    projection_completed_at DATETIME(3) NULL,
    active_conversation_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin
        GENERATED ALWAYS AS (
            CASE
                WHEN parent_run_id IS NULL
                    AND status IN (
                        'RUNNING',
                        'WAITING_CONFIRMATION',
                        'WAITING_EXTERNAL',
                        'CANCEL_REQUESTED')
                THEN conversation_id
            END
        ) STORED,
    create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    update_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
        ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uk_agent_run_id(run_id),
    UNIQUE KEY uk_agent_run_active(active_conversation_id),
    UNIQUE KEY uk_agent_run_parent_tool(parent_run_id, parent_tool_call_id),
    KEY idx_agent_run_conversation_status(conversation_id, status, id),
    KEY idx_agent_run_parent_status(parent_run_id, status, id),
    KEY idx_agent_run_user_status(user_id, status, update_time),
    KEY idx_agent_run_lease(status, lease_until),
    KEY idx_agent_run_status_deadline(status, deadline_at, id),
    KEY idx_agent_run_cancel(status, cancel_next_attempt_at),
    CONSTRAINT chk_agent_run_parent_identity CHECK (
        (parent_run_id IS NULL AND parent_tool_call_id IS NULL)
        OR (parent_run_id IS NOT NULL
            AND parent_tool_call_id IS NOT NULL
            AND agent_name IS NOT NULL)),
    CONSTRAINT chk_agent_run_status CHECK (
        status IN (
            'RUNNING',
            'WAITING_CONFIRMATION',
            'WAITING_EXTERNAL',
            'CANCEL_REQUESTED',
            'COMPLETED',
            'FAILED',
            'CANCELLED')),
    CONSTRAINT chk_agent_run_usage_settled CHECK (usage_settled IN (0, 1))
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'AgentScope V2 durable run';

CREATE TABLE afv_agent_event (
    id BIGINT NOT NULL AUTO_INCREMENT,
    run_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    sequence_no BIGINT NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    raw_event_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    raw_event_type VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    source VARCHAR(255) NULL,
    reply_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    block_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    tool_call_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    parent_tool_call_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    agent_name VARCHAR(128) NULL,
    output_type VARCHAR(32) CHARACTER SET ascii COLLATE ascii_bin NULL,
    payload_json MEDIUMTEXT NOT NULL,
    event_created_at DATETIME(3) NULL,
    redis_published_at DATETIME(3) NULL,
    publish_required TINYINT NOT NULL,
    publish_status VARCHAR(16) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    publish_claim_owner VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    publish_claim_until DATETIME(3) NULL,
    next_publish_attempt_at DATETIME(3) NULL,
    last_publish_error VARCHAR(1024) NULL,
    publish_attempts INT NOT NULL DEFAULT 0,
    create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uk_agent_event_sequence(run_id, sequence_no),
    KEY idx_agent_event_projection(run_id, output_type, sequence_no),
    KEY idx_agent_event_publish(publish_status, next_publish_attempt_at, id),
    KEY idx_agent_event_raw(run_id, raw_event_id),
    CONSTRAINT chk_agent_event_publish_state CHECK (
        (publish_required = 0 AND publish_status = 'NOT_REQUIRED')
        OR (publish_required = 1
            AND publish_status IN ('PENDING', 'CLAIMED', 'PUBLISHED'))),
    CONSTRAINT chk_agent_event_publish_attempts CHECK (publish_attempts >= 0),
    CONSTRAINT chk_agent_event_sequence CHECK (sequence_no >= 1),
    CONSTRAINT chk_agent_event_schema_version CHECK (schema_version >= 1)
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'AgentScope V2 committed event journal';

CREATE TABLE afv_agent_model_call_usage (
    id BIGINT NOT NULL AUTO_INCREMENT,
    run_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    model_call_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    provider VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    model_code VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    status VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    input_tokens BIGINT NULL,
    output_tokens BIGINT NULL,
    reasoning_tokens BIGINT NULL,
    cache_tokens BIGINT NULL,
    usage_json MEDIUMTEXT NULL,
    settlement_status VARCHAR(24) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    settlement_attempts INT NOT NULL DEFAULT 0,
    next_settlement_attempt_at DATETIME(3) NULL,
    settlement_claim_owner VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    settlement_claim_until DATETIME(3) NULL,
    downstream_settlement_id VARCHAR(128) CHARACTER SET ascii COLLATE ascii_bin NULL,
    last_settlement_error VARCHAR(1024) NULL,
    started_at DATETIME(3) NOT NULL,
    finished_at DATETIME(3) NULL,
    create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
    update_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
        ON UPDATE CURRENT_TIMESTAMP(3),
    PRIMARY KEY (id),
    UNIQUE KEY uk_agent_usage_call(run_id, model_call_id),
    KEY idx_agent_usage_settlement(
        settlement_status, next_settlement_attempt_at, id),
    KEY idx_agent_usage_run_status(run_id, status, id),
    CONSTRAINT chk_agent_usage_status CHECK (
        status IN ('STARTED', 'COMPLETED', 'FAILED', 'CANCELLED')),
    CONSTRAINT chk_agent_usage_settlement CHECK (
        settlement_status IN ('PENDING', 'CLAIMED', 'SETTLED')),
    CONSTRAINT chk_agent_usage_settlement_attempts CHECK (settlement_attempts >= 0),
    CONSTRAINT chk_agent_usage_tokens CHECK (
        (input_tokens IS NULL OR input_tokens >= 0)
        AND (output_tokens IS NULL OR output_tokens >= 0)
        AND (reasoning_tokens IS NULL OR reasoning_tokens >= 0)
        AND (cache_tokens IS NULL OR cache_tokens >= 0))
) ENGINE = InnoDB
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = 'AgentScope V2 model-call usage ledger';
