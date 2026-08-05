# v1.1.1 项目名称与时区修复完成总结

## 最终范围

- 创建 `v-1.1.1` 开发分支。
- Maven 项目版本、SCM tag 和前端包版本更新为 `1.1.1`。
- 项目设置新增项目名称输入、变更检测、空白校验、256 字符长度限制和保存成功反馈。
- 保存项目名后刷新项目上下文；后端既有逻辑同步固定剧本和分镜标题。
- 应用在 Spring Bean 初始化前读取 `APP_TIME_ZONE`，默认采用 `Asia/Shanghai`，消除 Docker/JVM 默认 UTC 导致的新建项目时间早 8 小时问题。
- Hikari 为每个新 MySQL 连接执行会话时区初始化，Docker MySQL 也设置默认会话时区；`MYSQL_SESSION_TIME_ZONE` 默认采用 `+08:00`。
- `.env.example` 增加应用与 MySQL 会话时区配置说明。
- 新增不依赖数据库的配置加载、应用时区应用及非法配置单元测试。

## 验证结果

- `pnpm exec eslint 'app/(dashboard)/projects/[id]/settings/page.tsx' 'lib/api/project.ts'`：通过，无警告。
- `pnpm exec tsc --noEmit`：通过。
- `pnpm build`：通过，Next.js 生产构建完成。
- `./mvnw -Dtest=ApplicationTimeZoneInitializerTests,ProjectServiceTests test`：通过，时区配置与项目服务测试全部成功。
- `docker compose --env-file .env.example -f docker-compose.yml config --quiet`：通过。
- `docker compose --env-file .env.example -f docker-compose.build.yml config --quiet`：通过。
- `git diff --check`：通过。

## 遗留事项

- 未自动迁移旧版本在 UTC 环境中写入的无时区历史时间，原因见 `technical-debt.md`。
