# 对象越权漏洞核查与修复

完成日期：2026-10-23。

## 报告核实

[原始 HackMD 报告](https://hackmd.io/@hieuPenguinnn/ai-fusion-video-idor-bola-project-asset) 针对 v1.2.0，作者明确说明只做静态源码审查。报告给出的 `92a543629d847a466d53b10318906537bf25bf71` 在本地是 annotated tag 对象，实际提交为 `856b94fcc685b2ff86660a4376ce808ad719baf2`。

核对该版本的项目、资产和剧本控制器及服务后，确认直接使用用户提交的对象 ID、缺少对象权限校验的问题成立。当前开发工作区已采用 ResourceQueryService / ResourceCommandService / CanvasAccessPolicy / ResourceAccessPolicy，报告中的主要直接调用已被替换；本次通过真实 HTTP 证明其隔离有效，并修复仍可绕过隔离的团队管理入口及其他认证问题。

## 本次修复

| 问题 | 原行为及影响 | 修复 |
| --- | --- | --- |
| 团队管理越权 | 登录用户可读取别的团队、修改团队、添加或删除成员、修改角色；可间接获取项目与媒体权限 | 新增 TeamAccessPolicy，服务层实时校验；团队写操作在同一行锁内授权和写入；保护创建者；仅创建者或系统管理员可授予管理员角色，拒绝伪造 OWNER 及未知角色；公开注册固定加入普通成员 |
| 敏感配置泄漏 | API 配置详情/列表可向普通账户返回 API key、app secret、代理密码；系统配置任意键和全量读取可泄漏邮件等配置 | 配置管理读取统一要求系统管理员；普通用户只允许读取创作所需的 resource_base_url；两处前端调用同步改为该单值接口 |
| 禁用账户仍可用旧令牌 | Bearer 认证直接建立已认证身份，没有检查账户状态；刷新入口仍可能签发新令牌 | 校验账户状态及令牌绑定的用户 ID；拒绝禁用、删除或身份不一致的账户刷新；认证异常清除安全上下文，不吞掉任意系统异常 |
| 角色撤销不生效 | 禁用的角色仍映射为权限；角色变化及账户状态可能受旧缓存影响 | 授权只采用启用角色，认证实时读取账户/角色；展示用角色缓存单独保留并在角色修改时清理；当前用户信息也不再显示已停用角色 |
| 首次初始化竞争 | 并发初始化请求可同时通过“未初始化”判断 | 使用 MyBatis Mapper XML 锁定管理员角色行，在 READ_COMMITTED 事务内串行完成检查及创建；6 个并发请求验证只有一个管理员创建成功 |

未修改基础 UI 组件、历史发布脚本或数据库迁移；未引入 fallback。权限判断不缓存，普通展示数据仍按既有缓存机制处理。

## 验证

12 个相关测试类合计 **139 项独立测试通过，0 失败、0 错误、0 跳过**。其中新增 ObjectAuthorizationIT 的 9 项测试使用完整 Spring 应用、临时 MySQL/Redis 和真实 HTTP；外部模型供应商未被调用。

| 测试类 | 数量 | 保护的行为 |
| --- | ---: | --- |
| ObjectAuthorizationIT | 9 | 跨用户及跨团队读写拒绝、数据不变、只读角色、权限撤销后的缓存和重放、团队提权、敏感配置、禁用账户与角色、初始化竞争、正常注册 |
| ResourceCommandIT | 51 | 资源权限、版本、类型化所有权、事务、字段白名单、幂等重放及派生资源 |
| ProjectMembershipIT | 6 | 成员角色、撤销授权、审计与并发 |
| CanvasCollaborationIT | 21 | 画布协作、资源历史及相关权限边界 |
| GenerationExecutionWiringIT | 17 | 生成任务身份、范围约束、取消及执行隔离 |
| MediaResourceIT | 12 | 媒体所有权、访问凭据、生命周期及持久化 |
| ProviderMediaAuthorizationTests | 7 | 供应商媒体访问授权 |
| MediaSourceFetcherTests | 4 | 来源地址、重定向及本地路径校验 |
| MediaSecurityBoundaryTests | 1 | 媒体网关 GET/HEAD 凭据边界与写入认证 |
| TokenServiceTests | 3 | 会话刷新、注销与多会话隔离 |
| UserServiceTests | 5 | 初始化及公开注册契约 |
| TeamServiceTests | 3 | 团队切换与当前归属 |

最后一次改动后重跑 ObjectAuthorizationIT、UserServiceTests、TeamServiceTests、TokenServiceTests、MediaSecurityBoundaryTests 共 21 项，验证人为保留旧缓存时撤权仍生效。测试初轮因新样例使用旧的场景字段名失败，改为当前 sceneDescription 契约后全部通过；没有弱化断言。

前端类型检查 `pnpm exec tsc --noEmit`、两处修改文件的 ESLint、修改差异空白检查均通过。测试应用启动及资源回归完成 Flyway validate（65 项迁移校验），本轮没有新增或修改迁移。

## 交付边界

当前工作区已有的大量未提交重构全部保留。修复已落在工作区，未提交、发布或部署；线上旧版本需要部署修复版本才能生效。运行时验证覆盖本轮列出的权限边界，补充检查了 AI 会话/运行查询、资源历史和媒体访问凭据的实际调用链；不代表线上入侵取证或对整个代码库的无漏洞保证。

同团队共享和默认团队注册策略保持不变。旧版公开媒体 URL 的历史协议未在本轮改造；当前托管媒体的权限与凭据另有验证。部署后的凭据轮换及成员审计由实际线上情况决定，详见 technical-debt.md。
