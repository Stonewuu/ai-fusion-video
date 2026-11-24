ALTER TABLE afv_agent_run
    ADD COLUMN stream_protocol_version INT NOT NULL DEFAULT 1 COMMENT '消息流协议版本：1为数据库事件，2为Redis流' AFTER agent_type,
    ADD COLUMN stream_run_id VARCHAR(64) CHARACTER SET ascii COLLATE ascii_bin NULL COMMENT '所属展示流的根运行标识' AFTER stream_protocol_version,
    ADD INDEX idx_agent_run_stream (stream_run_id, id),
    ADD CONSTRAINT chk_agent_stream_protocol CHECK (stream_protocol_version IN (1, 2));

UPDATE afv_agent_run SET stream_run_id = run_id;
