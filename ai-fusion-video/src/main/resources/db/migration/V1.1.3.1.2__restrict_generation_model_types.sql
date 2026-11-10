-- 生成模型统一为文字、图像、视频、音频四类；保留旧记录和已生成的媒体。
UPDATE afv_ai_model SET status=0, default_model=FALSE WHERE model_type IN (5,6,7,8);
UPDATE afv_ai_model SET default_model=FALSE WHERE model_type=9
  AND EXISTS (SELECT 1 FROM (SELECT id FROM afv_ai_model WHERE model_type=4 AND default_model=TRUE AND deleted=FALSE) current_audio_default);
UPDATE afv_ai_model SET model_type=4 WHERE model_type=9;

UPDATE afv_google_model_task SET deleted=TRUE,
  state=IF(state IN ('QUEUED','RUNNING','WAITING','INGESTING'),'CANCELLED',state),
  lease_owner=NULL, lease_until=NULL
WHERE model_type NOT IN (4,9);
UPDATE afv_google_model_task SET
  model_snapshot_json=JSON_SET(model_snapshot_json,'$.modelType',4),
  input_json=JSON_REMOVE(input_json,'$.targetLanguage','$.dimensions'),
  result_json=JSON_REMOVE(result_json,'$.embedding'),
  output_json=IF(output_json IS NULL,NULL,
    JSON_OBJECT('operation','normalized','body',CAST(JSON_REMOVE(output_json,'$.embedding') AS CHAR)))
WHERE model_type IN (4,9);

RENAME TABLE afv_google_model_task TO afv_audio_generation_task;
ALTER TABLE afv_audio_generation_task
  DROP COLUMN model_type,
  DROP COLUMN provider_task_id,
  MODIFY COLUMN id varchar(36) NOT NULL COMMENT '音频任务标识',
  MODIFY COLUMN result_json json DEFAULT NULL COMMENT '生成文字和托管音频引用',
  RENAME INDEX uk_google_task_request TO uk_audio_task_request,
  RENAME INDEX idx_google_task_dispatch TO idx_audio_task_dispatch,
  RENAME INDEX idx_google_task_owner TO idx_audio_task_owner,
  RENAME INDEX idx_google_task_model TO idx_audio_task_model,
  COMMENT='音频生成任务';

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
SELECT 'AUDIO_GENERATION_TASK',id,NULL,'HISTORY',JSON_ARRAY(input_json,result_json) FROM afv_audio_generation_task;
