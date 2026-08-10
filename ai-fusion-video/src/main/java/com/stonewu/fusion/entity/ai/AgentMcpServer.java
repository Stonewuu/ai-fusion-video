package com.stonewu.fusion.entity.ai;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import com.stonewu.fusion.common.BaseEntity;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;

@TableName("afv_agent_mcp_server")
@Data
@EqualsAndHashCode(callSuper = true)
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentMcpServer extends BaseEntity {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long userId;
    private String name;
    private String transport;
    private String url;
    private String headersJson;
    private String queryParamsJson;
    private String enabledToolsJson;
    private String protocolVersionsJson;
    private Integer timeoutSeconds;
    private Integer initializationTimeoutSeconds;
    private Integer status;
    private String lastTestStatus;
    private String lastTestMessage;
}
