# 融光助手 Skill 与 MCP 配置

融光助手使用 AgentScope Java 2.0.0 的原生 Skill 仓库、MCP 客户端和 Harness 上下文压缩能力。基础配置位于 `ai-fusion-video/src/main/resources/application.yaml`。

## Skill

默认启用随应用打包的 `classpath:agentscope/skills` 仓库。每个 Skill 使用独立目录，入口文件名必须为 `SKILL.md`：

```text
skills/
└── my-skill/
    ├── SKILL.md
    ├── references/
    └── scripts/
```

`SKILL.md` 至少包含名称和描述：

```markdown
---
name: my-skill
description: 说明何时应该使用该 Skill。
---

# 使用说明

这里写助手加载 Skill 后需要遵循的步骤。
```

增加外部只读 Skill 仓库的配置示例：

```yaml
fusion:
  agentscope:
    v2:
      skills:
        enabled: true
        fail-fast: true
        repositories:
          bundled:
            location: classpath:agentscope/skills
          local:
            location: file:./config/agentscope/skills
            lazy: true
```

- `location` 支持 `classpath:`、`file:` 和普通文件系统路径。
- `lazy: true` 仅对文件系统仓库生效，适合包含较多参考文件的 Skill。
- 仓库按配置顺序合并；同名 Skill 以后配置的仓库为准。
- `fail-fast: true` 会在仓库路径无效时阻止应用启动；设为 `false` 时跳过不可用仓库。
- 外部仓库在每次会话调用时重新读取目录快照；修改已有文件或增加 Skill 不需要重启应用。

## MCP

MCP 默认关闭，配置服务器并设置 `AGENTSCOPE_MCP_ENABLED=true` 后启用。客户端在应用启动时连接并发现工具，工具 Schema 会进入 Agent Kernel 指纹；服务器配置或工具 Schema 改变后需要重启应用。

Streamable HTTP 示例：

```yaml
fusion:
  agentscope:
    v2:
      mcp:
        enabled: ${AGENTSCOPE_MCP_ENABLED:false}
        fail-fast: true
        servers:
          asset-service:
            enabled: true
            transport: http
            url: ${ASSET_MCP_URL}
            headers:
              Authorization: Bearer ${ASSET_MCP_TOKEN}
            query-params:
              tenant: ${ASSET_MCP_TENANT:default}
            enabled-tools:
              - search_assets
              - get_asset
            agent-types:
              - ai_assistant_agent
            protocol-versions:
              - "2024-11-05"
              - "2025-03-26"
            timeout: 120s
            initialization-timeout: 30s
```

stdio 示例：

```yaml
fusion:
  agentscope:
    v2:
      mcp:
        enabled: true
        servers:
          local-tools:
            transport: stdio
            command: python
            args:
              - -m
              - my_mcp_server
            env:
              MCP_DATA_DIR: ${MCP_DATA_DIR:./data/mcp}
            enabled-tools:
              - inspect_media
            agent-types:
              - ai_assistant_agent
```

还支持 `transport: sse`。`enabled-tools` 为空时注册服务器发现的全部工具；`agent-types` 为空时对所有启用了工具的 Agent 开放。建议生产环境始终配置这两个白名单。`protocol-versions` 为空时使用 AgentScope/MCP SDK 默认协议版本；服务器协商新版协议时按示例显式声明。不同 MCP 服务器、平台工具和子 Agent 工具不能使用相同名称，冲突会阻止 Kernel 创建。

密钥只通过环境变量注入，不要直接提交到 YAML。`fail-fast: false` 可让单个不可用服务器被跳过，但对应工具不会进入本次启动的 Kernel。

## 默认上下文压缩

Harness 现在使用 AgentScope 2.0.0 的默认动态压缩策略：

- 模型提供上下文窗口时，在“上下文窗口减去 20,000 个保留 token”处触发；无法识别窗口时使用 160,000 token，并保留 50 条消息的兜底触发条件。
- 压缩后动态保留最近约 25% 的可用上下文，最少 2,000、最多 8,000 token；无法计算时保留最近 20 条消息。
- 汇总前执行默认的内存刷新/卸载，并启用旧工具结果裁剪。

这是 AgentScope 的默认值，融光没有复制或覆盖阈值。升级 AgentScope 后可直接继承框架修正。
