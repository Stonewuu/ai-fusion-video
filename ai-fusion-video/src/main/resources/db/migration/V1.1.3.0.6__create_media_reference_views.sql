-- 根据当前业务内容和历史快照判断媒体是否仍被使用，不通过这些视图授予读取权限。

-- 完整保留已有历史锚点，固定结果快照同样参与媒体保留检查。
CREATE SQL SECURITY INVOKER VIEW afv_media_reference_document AS
SELECT 'CANVAS_NODE' AS anchor_kind, id AS anchor_id, canvas_id,
       IF(deleted, 'HISTORY', 'LIVE') AS retention_class, config_json AS content FROM afv_canvas_node
UNION ALL
SELECT 'CANVAS_COMMAND', id, canvas_id, 'HISTORY', JSON_ARRAY(request_json, result_json, undo_json) FROM afv_canvas_operation
UNION ALL
SELECT 'CANVAS_REVISION', id, canvas_id, 'HISTORY', snapshot_json FROM afv_canvas_revision
UNION ALL
SELECT 'RESOURCE_COMMAND', id, canvas_id, 'HISTORY', JSON_ARRAY(request_json, result_json, undo_json)
FROM afv_resource_command WHERE committed_at IS NOT NULL
UNION ALL
SELECT 'RUN_PLAN', id, canvas_id, 'ACTIVE_REQUEST', snapshot_json FROM afv_canvas_run_plan WHERE expires_at > UTC_TIMESTAMP(6)
UNION ALL
SELECT 'CANVAS_RUN', id, canvas_id, 'HISTORY', snapshot_json FROM afv_canvas_run
UNION ALL
SELECT 'NODE_RUN', n.id, r.canvas_id, 'HISTORY', JSON_ARRAY(n.plan_json, n.input_json, n.output_json, n.checkpoint_json)
FROM afv_canvas_node_run n JOIN afv_canvas_run r ON r.id = n.run_id
UNION ALL
SELECT 'MEDIA_JOB', id, CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(context_json, '$.canvasId')), 'null') AS CHAR(36) CHARACTER SET ascii),
    'HISTORY', JSON_ARRAY(input_json, checkpoint_json) FROM afv_media_job
UNION ALL
SELECT 'GENERATION_JOB', j.id, CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(s.context_json, '$.canvasId')), 'null') AS CHAR(36) CHARACTER SET ascii),
    'HISTORY', JSON_ARRAY(j.input_json, j.result_json)
FROM afv_remote_generation_job j JOIN afv_generation_submission s ON s.id = j.submission_id
UNION ALL
SELECT 'TEXT_JOB', t.id, r.canvas_id, 'HISTORY', JSON_ARRAY(t.input_json, t.result_json)
FROM afv_canvas_text_job t JOIN afv_canvas_run r ON r.id = t.run_id
UNION ALL
SELECT 'RESOURCE_REVISION', CONCAT(v.command_id, ':', v.resource_type, ':', v.resource_id), c.canvas_id, 'HISTORY',
    JSON_ARRAY(v.before_json, v.after_json) FROM afv_resource_revision v JOIN afv_resource_command c ON c.id = v.command_id
UNION ALL
SELECT 'CANVAS_EXPORT', id, canvas_id, 'HISTORY', manifest_json FROM afv_canvas_export
UNION ALL
SELECT 'VALUE_SNAPSHOT', id, canvas_id, 'HISTORY', value_json FROM afv_canvas_value_snapshot;

-- 引用只决定保留，不授予文件读取权限。保留原有投影，并覆盖智能体与外置工作空间正文。
CREATE SQL SECURITY INVOKER VIEW afv_media_all_reference_document AS
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
    FROM afv_canvas_mcp_job j JOIN afv_canvas_run r ON r.id=j.run_id;

CREATE SQL SECURITY INVOKER VIEW afv_media_snapshot_reference AS
SELECT DISTINCT d.anchor_kind,d.anchor_id,d.canvas_id,d.retention_class,m.id AS media_id
FROM afv_media_all_reference_document d
JOIN JSON_TABLE(d.content,'$**.mediaId' COLUMNS(media_id LONGTEXT PATH '$' NULL ON EMPTY NULL ON ERROR)) v
JOIN afv_media_resource m ON m.id=v.media_id
UNION
SELECT d.anchor_kind,d.anchor_id,d.canvas_id,d.retention_class,m.id
FROM afv_media_all_reference_document d
JOIN JSON_TABLE(d.content,'$**.resourceId' COLUMNS(media_id LONGTEXT PATH '$' NULL ON EMPTY NULL ON ERROR)) v
JOIN afv_media_resource m ON m.id=v.media_id
UNION
SELECT d.anchor_kind,d.anchor_id,d.canvas_id,d.retention_class,m.id
FROM afv_media_all_reference_document d
JOIN JSON_TABLE(d.content,'$**.*' COLUMNS(media_uri LONGTEXT PATH '$' NULL ON EMPTY NULL ON ERROR)) v
JOIN afv_media_resource m ON CONCAT('media://',m.id)=v.media_uri
UNION
SELECT d.anchor_kind,d.anchor_id,d.canvas_id,d.retention_class,m.id
FROM afv_media_all_reference_document d
JOIN JSON_TABLE(d.content,'$**[*]' COLUMNS(media_uri LONGTEXT PATH '$' NULL ON EMPTY NULL ON ERROR)) v
JOIN afv_media_resource m ON CONCAT('media://',m.id)=v.media_uri
UNION
-- 智能体正文和 MCP 参数可能在字符串中嵌入 media://，不能只识别整个字段恰好等于引用的情况。
SELECT d.anchor_kind,d.anchor_id,d.canvas_id,d.retention_class,m.id
FROM afv_media_all_reference_document d JOIN afv_media_resource m
    ON JSON_SEARCH(d.content,'one',CONCAT('%media://',m.id,'%')) IS NOT NULL;

-- 来源映射、当前业务绑定、预设、派生物、活动票据与快照统一为媒体保留锚点。
-- 归档、删除节点或撤销命令不删除历史锚点。过期的未受理计划与票据不再保留。
CREATE SQL SECURITY INVOKER VIEW afv_media_reference AS
SELECT media_id, anchor_kind, anchor_id, canvas_id, retention_class
FROM afv_media_snapshot_reference
UNION
SELECT resource_id, 'BUSINESS_SOURCE', CONCAT(source_type, ':', source_id, ':', source_version, ':', source_field), NULL, 'HISTORY'
FROM afv_media_source
UNION
SELECT media_id, 'BUSINESS_BINDING', CONCAT(resource_type, ':', resource_id, ':', field_name), NULL, 'LIVE'
FROM afv_media_binding
UNION
SELECT media_id, 'PRESET', preset_key, NULL, 'LIVE'
FROM afv_media_preset_binding
UNION
SELECT source_resource_id, 'PREVIEW_SOURCE', job_id, NULL, 'DERIVATIVE'
FROM afv_media_preview_job
UNION
SELECT output_resource_id, 'PREVIEW_OUTPUT', job_id, NULL, 'DERIVATIVE'
FROM afv_media_preview_job WHERE output_resource_id IS NOT NULL
UNION
SELECT resource_id, 'ACCESS_TICKET', token_hash, NULL, 'ACTIVE_REQUEST'
FROM afv_media_access_ticket WHERE revoked = FALSE AND expires_at > UTC_TIMESTAMP(6)
UNION
SELECT m.id, 'GENERATION_INPUT', j.id, CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(s.context_json, '$.canvasId')), 'null') AS CHAR(36) CHARACTER SET ascii), 'HISTORY'
FROM afv_remote_generation_job j
JOIN afv_generation_submission s ON s.id = j.submission_id
JOIN JSON_TABLE(JSON_KEYS(j.input_json, '$.inputMediaVersions'), '$[*]'
    COLUMNS(media_id LONGTEXT PATH '$' NULL ON EMPTY NULL ON ERROR)) v
JOIN afv_media_resource m ON m.id = v.media_id
UNION
SELECT m.id, 'MEDIA_JOB_OUTPUT', j.id, CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(j.context_json, '$.canvasId')), 'null') AS CHAR(36) CHARACTER SET ascii), 'HISTORY'
FROM afv_media_job j JOIN afv_media_resource m
    ON m.created_by = j.actor_user_id AND m.request_id LIKE CONCAT('job:', j.id, ':%')
UNION
SELECT m.id, 'GENERATION_OUTPUT', s.id, CAST(NULLIF(JSON_UNQUOTE(JSON_EXTRACT(s.context_json, '$.canvasId')), 'null') AS CHAR(36) CHARACTER SET ascii), 'HISTORY'
FROM afv_generation_submission s JOIN afv_media_resource m
    ON m.created_by = s.actor_user_id AND m.request_id LIKE CONCAT('generation:', LOWER(s.kind), ':', s.task_id, ':%');
