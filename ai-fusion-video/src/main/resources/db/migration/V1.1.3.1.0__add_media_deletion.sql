-- 素材库移除与永久删除分别登记；已有版本快照不因清理素材库而失效。
CREATE TABLE afv_media_deletion (
    media_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '被移除的媒体标识',
    mode VARCHAR(16) NOT NULL COMMENT '删除方式：仅素材库或永久删除',
    request_id VARCHAR(96) NOT NULL COMMENT '本次操作请求标识',
    request_hash CHAR(64) CHARACTER SET ascii NOT NULL COMMENT '操作内容与身份摘要',
    actor_user_id BIGINT NOT NULL COMMENT '操作用户编号',
    actor_team_id BIGINT NULL COMMENT '操作时的团队作用域',
    resource_version INT NOT NULL COMMENT '操作前的媒体版本',
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '首次移除时间',
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6) COMMENT '最近操作时间',
    PRIMARY KEY (media_id)
) ENGINE=InnoDB COMMENT='媒体素材删除记录';

-- 文件删除在数据库事务外进行；持久化待处理记录支持服务重启及存储故障后的重试。
CREATE TABLE afv_media_purge (
    blob_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '待删除的存储文件标识',
    media_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL COMMENT '发起永久删除的媒体标识',
    state VARCHAR(16) NOT NULL DEFAULT 'PENDING' COMMENT '文件清理状态',
    attempts INT NOT NULL DEFAULT 0 COMMENT '文件清理失败次数',
    next_attempt_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) COMMENT '下次允许重试时间',
    updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6) COMMENT '最近状态变化时间',
    PRIMARY KEY (blob_id),
    KEY idx_media_purge_pending (state, next_attempt_at),
    KEY idx_media_purge_resource (media_id)
) ENGINE=InnoDB COMMENT='媒体永久删除文件清理任务';
