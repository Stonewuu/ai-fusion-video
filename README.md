<p align="center">
  <img src="assets/logo.png" alt="融光 Logo" width="368" />
</p>

<p align="center">
  <strong>简体中文</strong> | <a href="README_EN.md">English</a>
</p>

<h1 align="center">融光</h1>

<p align="center">Agent 驱动的视频创作平台</p>

<p align="center">
  <a href="https://github.com/Stonewuu/ai-fusion-video/releases"><img src="https://img.shields.io/github/v/release/Stonewuu/ai-fusion-video?display_name=tag" alt="GitHub Release" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Stonewuu/ai-fusion-video" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/Java-21-ED8B00" alt="Java 21" />
  <img src="https://img.shields.io/badge/Next.js-16-000000" alt="Next.js 16" />
</p>

<p align="center">
  <img src="https://stonewuu.github.io/Stonewuu/star-history/Stonewuu/ai-fusion-video.svg" alt="GitHub Star History" width="920" />
</p>

<table align="center" style="margin: 0 auto; display: table;">
  <tr>
    <td align="center">
      <b>想让你的品牌出现在这里？<a href="mailto:me@stonewu.com">联系我</a></b>
    </td>
  </tr>
</table>

<table align="center">
  <thead>
    <tr>
      <th align="center">联系作者</th>
      <th align="center">加入交流群</th>
      <th align="center">请作者喝杯咖啡</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center" width="33%">
        <img src="https://image.stonewu.com/file/wechat-qrcode.png" alt="作者微信二维码" width="220" />
      </td>
      <td align="center" width="33%">
        <img src="https://image.stonewu.com/file/ai-fusion-video-qrcode-group.png" alt="交流群二维码" width="220" />
      </td>
      <td align="center" width="33%">
        <img src="https://image.stonewu.com/file/alipay_qrcode.png" alt="打赏" width="220" />
      </td>
    </tr>
  </tbody>
</table>

<p align="center">
  <sub><b>微信群超过 200 人后无法扫码加入。请添加作者微信并备注“融光”。</b></sub>
</p>

<hr/>

融光是一款面向视频创作者的 Agent 驱动创作平台，将项目、剧本、分镜、素材以及图片与视频生成整合在统一工作区。创作者可以按分集和场景组织剧本、拆解与调整分镜，并围绕每个镜头完成素材生成和管理。

本仓库包含 Java 后端、Next.js 前端和 Docker Compose 部署配置。首次启动后，按照初始化向导创建管理员账号即可开始使用。

## 功能

| 模块 | 当前能力 |
| --- | --- |
| 项目与团队 | 管理创作项目、项目成员和协作角色 |
| 剧本创作 | 按分集和场景组织剧本，并将项目上下文交给 Agent |
| 分镜制作 | 从剧本整理分镜，编辑镜头内容、参考素材和生成结果 |
| 图片与视频 | 根据文本或参考素材生成图片与视频，并持续跟踪后台任务进度 |
| 素材管理 | 统一管理项目素材和可复用的图片、视频资源 |
| Agent 工作区 | 支持流式对话、多模态上下文、工具权限、Skill、MCP 与子 Agent |
| 模型与存储 | 集中配置文本、图片和视频模型，并选择本地或对象存储 |
| 系统管理 | 提供初始化向导、用户与角色管理、密码重置和版本检查 |

### Agent 与工具

内置 Agent 工作区可以结合项目内容持续对话，接收图片、视频、音频和文件等多模态输入，并在界面中展示推理过程、工具调用和任务状态。

Agent 运行基于 AgentScope，支持 Skill、MCP 和子 Agent。会话与运行状态会持久化保存，模型能力和工具权限可按会话配置，适合处理需要多步骤协作的创作任务。

### 模型与生成协议

模型和 API 接入信息统一在设置页管理，无需将密钥写入源码。对话与 Agent 支持 OpenAI 兼容接口、Anthropic、Gemini、DashScope 和 Ollama 等模型服务；图片与视频生成支持接入多种服务，并按能力在项目中选用相应模型。

具体可用模型取决于服务商账号、接口地址和模型权限。

### 存储

媒体素材可以保存在本地磁盘或 S3 兼容对象存储中。Agent 工作区同样支持数据库、本地磁盘和对象存储，可根据本地开发、单机部署或云端部署的需要灵活选择。

## 界面演示

https://github.com/user-attachments/assets/fe71cbb8-f9d9-4351-9a4c-cb8a0a6af7ba

https://github.com/user-attachments/assets/24d9443b-e463-405c-8ada-108236b4d6c2

https://github.com/user-attachments/assets/2f1de26c-5cd5-4be3-ad2e-81be2edd6956

https://github.com/user-attachments/assets/acd26ede-8b77-48c0-91dc-c80c5ed7ceca

https://github.com/user-attachments/assets/8a8ce3cf-4bf8-4f76-ad7c-0af373d16a5b

https://github.com/user-attachments/assets/be99d4c1-dc09-4616-8fba-06cb959c84c8

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Next.js 16、React 19、TypeScript、Tailwind CSS 4、Base UI / Shadcn、Zustand |
| 后端 | Java 21、Spring Boot 3.5、Spring Security、Spring AI、AgentScope、MyBatis-Plus |
| 数据 | MySQL 8、Redis、Flyway |
| 媒体 | FFmpeg、FFprobe、本地存储或 S3 兼容对象存储 |
| 部署 | Docker Compose、Nginx |

## 快速部署

### Docker Compose

推荐使用 Docker Compose 部署。安装 Docker Engine 和 Docker Compose 后，执行：

```bash
git clone https://github.com/Stonewuu/ai-fusion-video.git
cd ai-fusion-video

cp .env.example .env

docker compose up -d
```

如需自定义端口、密码或部署方式，请以 `.env.example` 为模板创建根目录 `.env`。该文件已被 Git 忽略，更新项目时无需修改 `docker-compose.yml`。

启动完成后访问 <http://localhost:8080>，按照页面提示创建管理员账号。查看服务状态或跟踪后端日志，可以运行：

```bash
docker compose ps
docker compose logs -f backend
```

Compose 默认启动 MySQL、Redis、后端、前端和 Nginx。Nginx 作为统一访问入口，并将 `/api/**` 和 `/media/**` 请求转发到后端。公网部署前，请通过 `.env` 修改默认数据库与 Redis 密码。

### 本地构建镜像

如需从当前源码构建前后端镜像，执行：

```bash
docker compose -f docker-compose.build.yml up -d --build
```

### 前后端独立部署

如果前端与后端分别使用不同的公网域名，请在 `.env` 中配置：

```env
PUBLIC_API_URL=https://api.example.com
CORS_ALLOWED_ORIGIN_PATTERNS=https://app.example.com
FRONTEND_PORT=3000
BACKEND_PORT=18080
```

`PUBLIC_API_URL` 应填写后端根地址，末尾不要添加 `/api`。后端默认全局允许跨域；生产环境建议像示例一样将 `CORS_ALLOWED_ORIGIN_PATTERNS` 限制为实际前端来源，多个来源使用英文逗号分隔。

使用预构建镜像：

```bash
docker compose -f docker-compose.yml -f docker-compose.separated.yml up -d
```

从源码构建镜像：

```bash
docker compose -f docker-compose.build.yml -f docker-compose.separated.yml up -d --build
```

前后端独立部署不会启动仓库内置的 Nginx。请分别为前端和后端配置 HTTPS 反向代理，并在“系统设置 > 通用”中确认站点公网地址和后端资源公网地址。

## 源码开发

### 环境要求

- JDK 21
- Node.js 20 和 pnpm 10
- Docker，用于启动本地 MySQL 与 Redis
- FFmpeg 和 FFprobe，用于视频合成与媒体探测

### 启动依赖和后端

```bash
cd ai-fusion-video
docker compose -f docker-compose-middleware.yml up -d
./mvnw spring-boot:run
```

Windows 环境可以运行 `.\mvnw.cmd spring-boot:run`。

后端默认使用 `local` 配置，相关设置位于 `ai-fusion-video/src/main/resources/application-local.yaml`。如果 FFmpeg 和 FFprobe 不在默认路径，请通过 `VIDEO_COMPOSE_FFMPEG_PATH` 和 `VIDEO_COMPOSE_FFPROBE_PATH` 指定实际位置。

### 启动前端

另开一个终端，在仓库根目录执行：

```bash
cd ai-fusion-video-web
pnpm install
pnpm dev
```

前端开发服务器会读取 `ai-fusion-video-web/.env.development`，并将 `/api/**` 和 `/media/**` 代理到 `http://localhost:18080`。如果后端监听其他端口，可以在 `.env.local` 中覆盖代理地址：

```env
DEV_BACKEND_URL=http://localhost:18081
```

### 本地地址

| 服务 | 地址 |
| --- | --- |
| 前端 | <http://localhost:3000> |
| 后端 | <http://localhost:18080> |
| Swagger UI | <http://localhost:18080/swagger-ui.html> |
| MySQL | `localhost:43306` |
| Redis | `localhost:46379` |

## 配置说明

### Docker 环境变量

Docker Compose 会自动读取仓库根目录的 `.env`。不同启动方式使用的配置如下：

| 部署方式 | Compose 文件 | 需要关注的变量 |
| --- | --- | --- |
| 默认统一入口 | `docker-compose.yml` | 必须关注 MySQL root 密码和 Redis 密码；应用账号可选；可按需修改 `MYSQL_DATABASE`、`APP_PORT`、`JAVA_OPTS` 和 CORS 白名单；`PUBLIC_API_URL` 保持为空 |
| 本地构建统一入口 | `docker-compose.build.yml` | 与默认统一入口相同，仅镜像改为本地构建 |
| 前后端独立部署 | 基础 Compose 文件 + `docker-compose.separated.yml` | 必须填写 `PUBLIC_API_URL`；生产环境建议填写 `CORS_ALLOWED_ORIGIN_PATTERNS`；可按需修改 `FRONTEND_PORT`、`BACKEND_PORT`；`APP_PORT` 不生效 |

公网部署前必须修改 MySQL root 密码和 Redis 密码；如启用 MySQL 应用账号，也必须设置独立密码。`FRONTEND_PORT`、`BACKEND_PORT` 和 `PUBLIC_API_URL` 仅在组合使用 `docker-compose.separated.yml` 时生效；`CORS_ALLOWED_ORIGIN_PATTERNS` 对所有部署方式生效。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MYSQL_ROOT_PASSWORD` | `123456` | MySQL `root` 管理密码，公网部署前必须修改 |
| `REDIS_PASSWORD` | `123456` | Redis 访问密码，公网部署前必须修改 |
| `MYSQL_DATABASE` | `ai_fusion_video` | MySQL 数据库名，通常无需修改 |
| `APP_PORT` | `8080` | 默认统一入口端口，启动后访问 `http://localhost:8080` |
| `FRONTEND_PORT` | `3000` | 前后端独立部署时暴露的前端端口 |
| `BACKEND_PORT` | `18080` | 前后端独立部署时暴露的后端端口 |
| `PUBLIC_API_URL` | 空 | 前后端独立部署时的后端公网根地址，不包含 `/api` |
| `CORS_ALLOWED_ORIGIN_PATTERNS` | `*` | 允许访问后端的浏览器来源；`*` 表示全部，生产环境建议配置明确来源，多个来源用英文逗号分隔 |
| `JAVA_OPTS` | `-Xms512m -Xmx1024m` | 后端 JVM 参数，通常无需修改 |
| `MYSQL_USERNAME` | 空 | 可选的 MySQL 应用账号，不能设置为 `root`；留空时沿用 root 账号 |
| `MYSQL_PASSWORD` | 空 | 可选的 MySQL 应用账号密码，必须与 `MYSQL_USERNAME` 同时配置 |

使用统一入口部署时，`PUBLIC_API_URL` 必须保持为空；`CORS_ALLOWED_ORIGIN_PATTERNS` 可保留 `*`，也可限制为站点的实际来源。

`MYSQL_USERNAME` 和 `MYSQL_PASSWORD` 必须同时配置或同时留空。直接复制 `.env.example` 时，这两个变量为空，后端仍使用 `root` 和 `MYSQL_ROOT_PASSWORD`，与历史版本默认行为一致。对于全新数据库，如在首次启动前填写这两个变量，MySQL 会自动创建普通账号并授予 `MYSQL_DATABASE` 的访问权限。已有数据库如需改用普通账号，请先在 MySQL 中创建用户并授权，再填写这两个变量；修改 `.env` 不会自动变更已有数据库账号。

默认部署不会向宿主机发布 MySQL 或 Redis 端口。如需使用本机工具直接访问，请在所用的 Compose 文件中取消对应 `ports` 配置的注释；示例端口仅绑定到宿主机 `127.0.0.1`。

### 公网资源地址

云端图片或视频模型通常无法读取 `localhost` 或局域网中的参考素材。请使用公网对象存储，或将后端资源端点安全地暴露到公网，然后在“系统设置 > 通用”中保存对应的公网地址。

站点公网地址用于密码重置邮件和页面链接，后端资源公网地址用于媒体链接和 Agent 引用；两者可以配置为不同域名。

## 项目结构

```text
.
├─ ai-fusion-video/       Java 后端、Flyway 迁移与后端测试
├─ ai-fusion-video-web/   Next.js 前端
├─ docker/                Nginx 配置
├─ docker-compose.yml     预构建镜像部署
├─ docker-compose.build.yml
└─ docker-compose.separated.yml
```

## 检查与测试

后端：

```bash
cd ai-fusion-video
./mvnw test
./mvnw -Dtest=FlywayMigrationNamingTests test
./mvnw -Pagentscope-integration verify
```

前端：

```bash
cd ai-fusion-video-web
pnpm lint
pnpm build
```

Flyway 迁移脚本位于 `ai-fusion-video/src/main/resources/db/migration/`。版本格式、基线和历史迁移约束见 [Flyway 迁移说明](ai-fusion-video/src/main/resources/db/migration/README.md)。

## 参与贡献

欢迎提交问题反馈、功能建议和代码改进。对于影响范围较大的功能或架构调整，建议先创建 Issue 说明方案，确认方向后再开始开发。

### 开发流程

1. Fork 本仓库并克隆到本地，然后将本仓库添加为 `upstream`：

   ```bash
   git clone https://github.com/YOUR-USERNAME/ai-fusion-video.git
   cd ai-fusion-video
   git remote add upstream https://github.com/Stonewuu/ai-fusion-video.git
   ```

2. 从最新的 `main` 创建功能分支，请勿直接在 `main` 或版本开发分支上修改：

   ```bash
   git fetch upstream
   git checkout main
   git pull --ff-only upstream main
   git checkout -b feat/your-change
   ```

   分支名称建议使用 `feat/`、`fix/` 或 `docs/` 前缀，并简要说明改动内容。

3. 完成开发后，根据改动范围运行检查：

   ```bash
   # 后端
   cd ai-fusion-video
   ./mvnw test

   # 前端
   cd ../ai-fusion-video-web
   pnpm lint
   pnpm build
   ```

4. 提交并推送个人分支：

   ```bash
   git add path/to/changed-file
   git commit -m "feat: describe your change"
   git push -u origin feat/your-change
   ```

5. 创建 Pull Request 时，将目标分支设置为本仓库的 `main` 分支。

提交前请确保改动范围清晰，不包含密钥、账号或本地环境文件；涉及行为变更时，请同步补充测试和必要文档。

## 致谢

- 感谢 [LinuxDo](https://linux.do) 社区对项目的支持
- 感谢 [waoowaoo](https://github.com/saturndec/waoowaoo) 提供的剧本界面设计参考

## License

[MIT License](LICENSE)

<p align="center">
  <sub>Built by <a href="https://github.com/Stonewuu">Stonewu</a></sub>
</p>
