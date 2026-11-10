CREATE TABLE afv_google_model_task (
  id varchar(36) NOT NULL COMMENT '模型任务标识',
  request_id varchar(80) NOT NULL COMMENT '客户端幂等请求标识',
  user_id bigint NOT NULL COMMENT '发起用户',
  team_id bigint DEFAULT NULL COMMENT '发起团队',
  model_id bigint NOT NULL COMMENT '模型配置标识',
  model_name varchar(255) NOT NULL COMMENT '模型显示名称',
  model_type int NOT NULL COMMENT '模型类型：4语音合成、5转写、6嵌入、7实时会话、8研究、9音乐',
  model_snapshot_json json NOT NULL COMMENT '模型参数快照，不包含渠道凭据',
  api_config_id bigint NOT NULL COMMENT '渠道配置标识',
  api_route_json json NOT NULL COMMENT '渠道地址、项目和鉴权方式快照，不包含凭据',
  input_json json NOT NULL COMMENT '输入文本、媒体引用和参数',
  input_hash varchar(64) NOT NULL COMMENT '输入摘要',
  state varchar(24) NOT NULL COMMENT '任务状态',
  provider_task_id varchar(512) DEFAULT NULL COMMENT '供应商任务标识',
  output_json json DEFAULT NULL COMMENT '已收到的模型输出，托管保存后清除',
  result_json json DEFAULT NULL COMMENT '文本、向量及托管音频引用',
  error_message varchar(1000) DEFAULT NULL COMMENT '用户可见错误说明',
  lease_owner varchar(80) DEFAULT NULL COMMENT '当前工作者',
  lease_until datetime(6) DEFAULT NULL COMMENT '执行租约到期时间',
  next_poll_at datetime(6) NOT NULL COMMENT '下次调度时间',
  deadline_at datetime(6) NOT NULL COMMENT '任务截止时间',
  create_time datetime(6) NOT NULL COMMENT '创建时间',
  update_time datetime(6) NOT NULL COMMENT '更新时间',
  deleted tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否已删除',
  PRIMARY KEY (id),
  UNIQUE KEY uk_google_task_request (user_id, request_id),
  KEY idx_google_task_dispatch (state, next_poll_at, lease_until),
  KEY idx_google_task_owner (user_id, team_id, create_time),
  KEY idx_google_task_model (model_id, state)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Google 多模态模型任务';

-- 模型任务的输入和生成结果参与引用检查。
CREATE OR REPLACE SQL SECURITY INVOKER VIEW afv_media_all_reference_document AS
SELECT CONVERT(anchor_kind USING utf8mb4) COLLATE utf8mb4_bin AS anchor_kind,
    CONVERT(anchor_id USING utf8mb4) COLLATE utf8mb4_bin AS anchor_id,
    CONVERT(canvas_id USING utf8mb4) COLLATE utf8mb4_bin AS canvas_id,
    CONVERT(retention_class USING utf8mb4) COLLATE utf8mb4_bin AS retention_class,content
FROM afv_media_reference_document
UNION ALL
SELECT 'AGENT_RUN',run_id,CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(canvas_scope_json,'$.canvasId')),'null') AS CHAR(36) CHARACTER SET ascii),
    'HISTORY',JSON_ARRAY(CAST(agent_definition_snapshot_json AS JSON),canvas_scope_json) FROM afv_agent_run
UNION ALL
SELECT 'AGENT_EVENT',CAST(e.id AS CHAR),CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(r.canvas_scope_json,'$.canvasId')),'null') AS CHAR(36) CHARACTER SET ascii),
    'HISTORY',CAST(e.payload_json AS JSON) FROM afv_agent_event e JOIN afv_agent_run r ON r.run_id=e.run_id
UNION ALL
SELECT 'AGENT_STATE',CONCAT(s.session_id,':',s.state_key,':',s.item_index),
    CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(r.canvas_scope_json,'$.canvasId')),'null') AS CHAR(36) CHARACTER SET ascii),
    'HISTORY',CAST(s.state_data AS JSON) FROM afv_agent_state s LEFT JOIN afv_agent_run r ON r.agent_state_session_id=s.session_id
UNION ALL
SELECT 'AGENT_MESSAGE',CAST(m.id AS CHAR),NULL,'HISTORY',
    JSON_OBJECT('content',m.content,'references',IF(JSON_VALID(m.references_json),CAST(m.references_json AS JSON),NULL),
        'referenceText',m.references_json) FROM afv_agent_message m
UNION ALL
SELECT 'AGENT_WORKSPACE',CAST(id AS CHAR),NULL,'LIVE',media_references_json FROM afv_agent_workspace_entry
    WHERE deleted=FALSE AND media_references_json IS NOT NULL
UNION ALL
SELECT 'CANVAS_AGENT_JOB',j.node_run_id,r.canvas_id,'HISTORY',JSON_ARRAY(j.snapshot_json,j.input_json,j.scope_json)
    FROM afv_canvas_agent_node_job j JOIN afv_canvas_run r ON r.id=j.canvas_run_id
UNION ALL
SELECT 'CANVAS_MCP_JOB',j.id,r.canvas_id,'HISTORY',JSON_ARRAY(j.input_json,j.result_json)
    FROM afv_canvas_mcp_job j JOIN afv_canvas_run r ON r.id=j.run_id
UNION ALL
SELECT 'GOOGLE_MODEL_TASK',id,NULL,'HISTORY',JSON_ARRAY(input_json,result_json) FROM afv_google_model_task;
