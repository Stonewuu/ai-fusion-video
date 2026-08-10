# ai-fusion-video-web

融光前端应用，负责项目管理、剧本与分镜编辑、素材管理、Agent Pipeline 可视化和系统设置等界面能力。

完整部署说明请参考仓库根目录的 [README](../README.md)。

## 技术栈

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS v4
- Zustand
- Axios

## 本地开发

```bash
pnpm install
pnpm dev
```

默认访问地址：<http://localhost:3000>

前端默认使用同源 `/api/**` 和 `/media/**`，Next.js 开发服务器会将请求代理到
<http://localhost:18080>。

代理目标在已提交的 `.env.development` 中显式配置。本地后端端口不同时，修改该文件或在
`.env.local` 中覆盖：

```env
DEV_BACKEND_URL=http://localhost:18081
```

## 生产构建

```bash
pnpm build
pnpm start
```

`next.config.ts` 已启用 `standalone` 输出，便于容器化部署。

Docker 镜像在容器启动时读取 `PUBLIC_API_URL` 并生成浏览器运行时配置：

- 留空：统一入口部署，浏览器使用同源 `/api`。
- `https://api.example.com`：前后端分域，浏览器直接访问该后端。

该地址不包含末尾 `/api`。后端默认允许所有来源跨域访问；生产环境建议配置
`CORS_ALLOWED_ORIGIN_PATTERNS=https://app.example.com` 限制实际前端来源。不要使用 `NEXT_PUBLIC_API_BASE_URL`，它属于
构建时变量，无法可靠配置已经发布的镜像。

## 目录说明

- `app/`：App Router 页面与布局
- `components/`：业务组件与通用 UI 组件
- `lib/api/`：后端接口封装
- `lib/store/`：客户端状态管理
- `proxy.ts`：页面访问控制与登录跳转逻辑
