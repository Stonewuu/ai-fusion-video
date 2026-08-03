# 融光助手 Skill 与 MCP 配置

融光助手使用 AgentScope Java 2.0.0 的原生 Skill 仓库、MCP 客户端和 Harness 上下文压缩能力。基础配置位于 `ai-fusion-video/src/main/resources/application.yaml`。

## Skill

用户可以在“系统设置 → 智能体配置 → 我的 Skills”直接创建和编辑 Skill。用户 Skill 按 AgentScope 官方工作空间结构保存到当前用户命名空间：

```text
agents/ai_assistant_agent/users/<userId>/skills/<name>/SKILL.md
```

这些 Skill 会与应用内置 Skill 合并，同名时用户版本优先；助手输入框输入 `/` 即可模糊检索并主动引用。选中的名称通过结构化 `enabledSkills` 字段发送，服务端校验当前用户确实可用后，将正文作为当前 Agent Kernel 的强制 Skill 指令；不会依赖普通上下文文本让模型自行猜测是否加载。单次最多主动引用 8 个 Skill，正文合计最多 128 KB。每个用户最多创建 64 个 Skill，单个正文最多 256 KB。工作空间迁移期间 Skill 编辑会暂时锁定，迁移完成或解除失败状态后自动恢复。

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

用户可以在“系统设置 → 智能体配置 → 我的 MCP”添加自己的 Streamable HTTP 或 SSE 服务。用户配置保存在数据库中，不随工作空间存储后端迁移；保存后可执行连接测试并查看发现的工具。用户 MCP 只允许公网 HTTP(S) 地址，会拒绝回环、链路本地和内网目标以降低 SSRF 风险。需要运行本机命令的 stdio MCP 仍由管理员通过服务端 YAML 配置。

用户 MCP 连接按用户和 Agent Kernel 隔离，Kernel 销毁时连接同步关闭。配置变更会生成新的配置指纹和 Kernel，避免继续使用旧 URL、请求头或密钥，也避免不同用户复用对方带凭据的 MCP 客户端。每个用户最多配置 32 个 MCP 服务、启用 256 个工具。

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

## 工作空间存储与迁移

“智能体配置”支持三种正文存储后端，元数据始终保存在 MySQL：

- `database`：默认模式，正文直接保存在工作空间条目表中，不依赖额外基础设施。
- `local`：正文保存在服务器目录；Docker 默认目录为 `/app/data/agent-workspace`。
- `object_storage`：复用“存储配置”中已启用的 S3 兼容配置，不重复保存 Endpoint 或密钥。

完整工作空间正文不会写入 Redis；Redis 只缓存小体积的 Skill 名称、描述等目录元数据，因此不会随着 Skill 正文和资源文件增长而线性占用 Redis 内存。缓存值统一使用带完整类型元数据的 JSON 契约，键前缀包含序列化 Schema 版本；升级契约时直接进入新键空间，不读取或兜底解析旧格式。

切换后端时系统先复制全部正文并校验 SHA-256 和字节数，再在数据库事务中切换条目引用与当前配置。旧正文和迁移明细会保留，因此最近一次成功迁移可以回滚。复制期间工作空间只读，助手仍可读取已有 Skill；迁移失败时管理员需要先检查错误，再在页面解除失败状态。

两份根目录 Docker Compose 已挂载独立持久卷：

```yaml
backend:
  volumes:
    - agent_workspace_data:/app/data/agent-workspace

volumes:
  agent_workspace_data:
```

普通 Docker 本地卷只适用于单后端实例。多主机或多副本部署应选择数据库、对象存储，或把该目录替换为所有实例共享的文件系统卷。
