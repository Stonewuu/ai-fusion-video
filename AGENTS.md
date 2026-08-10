# 项目协作规则

## 适用范围与优先级

- 本文件只记录跨模块、长期稳定的全局约束，不包含具体业务页面、路由或功能规则。
- 前端规则适用于 `ai-fusion-video-web/`；Java 规则适用于 `ai-fusion-video/`。
- 禁止写 fallback 代码，除非需求中出现兼容要求，产生的技术债务必须留档以便完整实现功能

## 代码组织

- 前端单文件不得超过 1000 行；页面建议不超过 500 行，业务组件建议不超过 400 行。接近上限时拆分，禁止用压缩格式或超长单行规避。
- 页面只负责路由参数、数据加载、状态编排和组件组合；列表、筛选栏、编辑器、表单、标签页、空状态、操作栏应独立成组件，页面不得内嵌多个完整表单或复杂列表项。
- 路由专用组件放当前路由的 `_components/`，跨页面组件放 `components/`；常量、类型映射、纯工具函数独立存放，不与大段 JSX 混写。

## 前端设计系统

### Shadcn 组件

- 在 `ai-fusion-video-web/` 中使用 `pnpm dlx shadcn@latest add <component> --yes` 安装，禁止使用 `npx`；组件从 `@/components/ui/<component>` 导入。
- 不修改 `components/ui/` 下的组件源码；差异通过使用处的 `className` 等公开接口实现。
- Select 基于 `@base-ui/react`（base-maia）：必须传 `items: { value, label }[]`；`SelectValue` 使用 `placeholder`，不传 children；结构为 `SelectContent > SelectGroup > SelectItem`。
- 调整 Select 字号时同时覆盖 `SelectTrigger`、`SelectContent`、`SelectItem`；不使用 `rounded-lg!` 等强制圆角覆盖，保留默认 `rounded-4xl`。
- 所有除了列表和表格的区域，设计必须要有主次性，禁止盒子平铺。

### 按钮

- `@/components/ui/button` 是业务按钮的唯一视觉基线；页面只可调整布局宽度和对齐，不得用 `className` 重写高度、内边距、圆角、字号、颜色、阴影或交互态。
- 只使用已有尺寸：`xs`/`icon-xs`=24px，`icon-sm`=32px，`default`/`sm`/`icon`=36px，`lg`/`icon-lg`=40px；同组按钮统一 `gap-2`（8px）。默认文字 14px、图标 16px，紧凑尺寸由组件自动缩小。
- 只使用已有语义：`default`=区域唯一主操作，`outline`=普通次操作，`secondary`=低强调/成组操作，`ghost`=工具栏/导航/轻操作，`destructive`/`destructive-ghost`=危险操作，`ai`/`image`/`frame`/`video`=对应 AI 能力，`link`=文本链接。禁止在页面复制这些配色或渐变。
- 文案使用简短动词；解释放 Tooltip 或 `title`。每个视觉区域仅一个主按钮；纯图标按钮必须有 `aria-label` 或 `title`；保留共享焦点环、按压、加载和禁用态。
- 原生 `<button>` 仅用于标签页、分段选择器、列表行和非业务图标控件，仍须遵守共享按钮的尺寸、圆角、焦点和禁用规范。

### 表面与毛玻璃

- 普通页面面板统一使用 `rounded-xl border border-border/30 bg-card/50 backdrop-blur-sm`；不需要透出背景时改用实体 `bg-card`，不得为追求效果强行毛玻璃化。
- 面板内嵌区域使用 `rounded-lg border border-border/20 bg-background/70`，不再叠加毛玻璃或阴影；浮层优先使用 Shadcn 组件，自定义浮层统一采用 `rounded-2xl border border-border/40 bg-popover/80 backdrop-blur-xl shadow-xl`。
- 毛玻璃必须同时具备半透明背景和细边框；同一视觉层级只保留一层模糊，禁止嵌套多层 `backdrop-blur-*`、大面积高强度模糊或随意组合透明度。
- 除图片/视频遮罩等必须使用黑白叠层的场景外，颜色统一使用 `background`、`card`、`popover`、`muted` 等语义令牌，避免 `bg-white/*`、`border-white/*` 等破坏明暗主题。

### 边框与圆角

- 默认边框为 1px：`border-border/20` 用于内部分隔，`/30` 用于普通表面，`/40` 用于浮层或可交互容器，`/50` 用于悬停/强调；主色和危险色边框只表达选中、焦点、校验或危险状态。
- `border-2`、虚线边框仅用于拖放上传、明确的空状态入口或强焦点反馈；禁止用多重边框装饰普通卡片。
- 圆角按层级统一：内嵌区域 `rounded-lg`，普通面板与按钮 `rounded-xl`，Dialog/Popover 等浮层 `rounded-2xl`。`rounded-full` 仅用于头像、徽标、圆形图标按钮和胶囊标签；不得用于普通输入框、卡片或表单。

### 间距

- 全局使用 4px 网格和 Tailwind 标准刻度，禁止无必要的任意值。控件内部用 `gap-1`/`gap-1.5`（4/6px），紧密关联项用 `gap-2`/`gap-3`（8/12px），区块间用 `gap-4`/`gap-6`（16/24px）。
- 面板内边距：密集内容 `p-3`，默认 `p-4`，页面主区或 Dialog 可用 `p-5`/`p-6`；同一层级必须保持一致，不用空白、分割线和卡片嵌套重复制造层级。
- 同组按钮、字段、列表项的间距只使用一个档位；优先通过父容器 `gap-*` 管理，避免对子项散落设置 `mt-*`、`ml-*`。

### 阴影与层级

- 静态页面面板默认无阴影；可点击卡片或选中态最多使用 `shadow-sm`，悬停可提升一级到 `shadow-md`/`shadow-lg`，Popover/Dialog 等脱离文档流的浮层使用 `shadow-xl`/`shadow-2xl`。
- 彩色阴影仅由共享语义按钮或明确的主状态组件提供，透明度保持克制；禁止给普通卡片、大面积背景或多层嵌套容器添加彩色光晕。
- 单个元素只使用一个阴影层级；阴影用于表达高度，不替代边框、选中态或焦点环。悬停前后最多变化一个层级，避免突兀跳变。

### 动画与过渡

- 悬停、焦点、颜色、透明度和简单位移使用 CSS/Tailwind；仅在进出场、列表编排、布局变化或需要卸载动画时使用 Framer Motion。禁止同一属性同时由 CSS 与 Motion 驱动。
- 时长分三级：即时反馈 100–150ms，按钮/浮层/折叠 200–250ms，页面或大区块进场 300–500ms；除持续加载/进度动画外不得超过 500ms。
- 进入使用 `ease-out`，退出使用更短的 `ease-in`；项目级页面动画统一使用 `[0.25, 0.46, 0.45, 0.94]`。弹簧只用于直接操控、拖拽或明确的弹性反馈，不用于普通淡入淡出。
- 优先动画 `opacity` 和 `transform`，避免过渡宽高、边距、阴影模糊半径等触发布局/重绘的属性；折叠内容确需高度动画时必须限制溢出并保持锚点稳定。
- 位移和缩放保持克制：控件按压交由共享组件处理；卡片悬停位移不超过 4px、缩放不超过 `1.01`；Dialog/Popover 进场缩放不低于 `0.95`。列表 stagger 间隔 40–80ms，总进场时间不超过 500ms。
- 禁止泛用 `transition-all`，按需使用 `transition-colors`、`transition-opacity`、`transition-transform` 或 `transition-shadow`；禁止用循环漂浮、闪烁、旋转等装饰动画干扰内容，加载、进度和骨架屏除外。
- Select、Menu、Popover、Tooltip 等沿用 Shadcn 的 `animate-in`/`animate-out`、淡入、缩放和方向滑入，禁止用 `animate-none` 关闭公共动画。
- 所有非必要动画必须支持 `prefers-reduced-motion`：CSS 使用 `motion-reduce:*`，Framer Motion 使用 `useReducedMotion` 或等价判断；降级后仍须保留完整状态反馈。

## 页面层级、状态与文案

- 页面按“标题 → 一级导航/目录 → 当前对象摘要 → 主要工作区”组织；同层级统一字号、间距和表面样式，避免所有区域等权重卡片化。
- 编辑区突出当前对象名称、状态和主操作，辅助信息降低字号与对比度。
- 每个视觉区域只有一个主操作；删除操作在颜色和位置上与主操作区分，并靠近被操作对象。
- 选中态使用位置、背景、边框、文字/图标颜色或轻阴影明确表达；圆角容器内不用无语义侧边竖条、悬浮色块或突兀装饰线。不得只依赖颜色，必须保留键盘焦点环。
- 同类列表、标签页、编辑器统一选中、悬停和焦点状态；横向标签页支持滚动、选中项居中和清晰选中态。
- 标题和字段标签优先承载信息；可由布局直接理解的内容不重复解释，必要说明放 Tooltip、`title` 或帮助图标；空状态仅保留状态和一个建议操作。

## 布局、滚动与浮层

- Dashboard 使用 `h-screen overflow-hidden flex flex-col`；Header、Sidebar 固定定位，主体使用 `flex-1 min-h-0 pt-20` 为 Header 留位。`main` 仅作为全宽的 `flex-1 min-h-0 overflow-hidden` 语义外壳，其直接子级使用覆盖式 ScrollArea 承担滚动；不在 `main` 上设置水平内边距或内容最大宽度，水平安全边距、居中和限宽由稳定的路由内容框负责。
- 桌面端主内容底部间距统一与 Sidebar 底部间距一致（当前为 `pb-3`），移动端为底部悬浮操作保留 `pb-20`。普通页由共享路由内容框承担该 padding，全高度工作区由页面根节点承担；不得依赖滚动容器自身 padding，也不得在子页面再叠加无业务意义的尾部 `mb-*` / `pb-*`。全高度工作区使用绝对定位的底部浮层时，`bottom` 必须与页面根节点的底部间距一致，底部安全区保持 `background` 语义背景，浮层阴影应在安全区内自然衰减；禁止用覆盖层截断阴影，也禁止用高不透明遮罩削弱毛玻璃的透景效果。
- 普通页面由 `main` 内的覆盖式 ScrollArea viewport 滚动，共享路由内容框使用自然增高的 `min-h-full shrink-0` DOM 结构；全高度页面沿父级 Flex 链使用 `grow basis-0 min-h-0` 填满剩余高度，其中 `basis-0` 必须是确定的 `0px`，避免 `flex-1` 的百分比 basis 在自然高度父级中退化为内容尺寸并被长内容撑开。禁止 `h-[calc(100vh-Xrem)]`，也禁止通过挂载后的 Hook、fallback 状态或子节点出现后再切换页面宽高。
- 同一模块的同级页面必须使用统一的主内容宽度和居中容器，不得各自设置不同的 `max-w-*`；宽度约束统一放在最近的共享 `layout.tsx`。设置模块统一使用共享内容框内的 `mx-auto w-full max-w-[1200px]`，子页面只负责自身内容结构。
- 普通内容页禁止在主列表、主卡片或页面根容器上再设置纵向 `max-h-* overflow-y-auto`，纵向滚动统一交给 `main` 内的覆盖式 ScrollArea viewport；Dialog、Sheet 和明确的独立栏位除外。
- 页面侧边滚动条必须使用“隐藏原生滚动条 + 绝对定位 Scrollbar/Thumb”的覆盖式结构，叠加在内容上方，不得占据布局宽度，也不得使用 `scrollbar-gutter` 预留空间；从无滚动条变为有滚动条时，内容宽度和居中位置必须保持不变。共享主 ScrollArea 必须以 `pathname` 作为 `key`，路由切换后不得继承上一页面的 `scrollTop`、overflow 或 Scrollbar 状态。
- 全高度工作区确需内部滚动时，使用“全宽覆盖式滚动外壳 + 内层水平安全边距 + 居中限宽内容”的 DOM 结构；滚动外壳必须直接占满公共内容区，使滚动条从首次渲染起固定在浏览器最右侧。禁止用运行时布局 Hook、fallback、负 margin、宽度过渡或子页面 `:has()` 反向改写父级 `main` 的宽度与 padding。
- `overflow-hidden` 容器内的菜单、Tooltip、Popover 等通过 `createPortal(..., document.body)` 渲染，并用触发器 `ref` + `getBoundingClientRect()` 定位到触发器下方，避免被裁切。
- 长 Dialog 在使用处设置 `max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden`；Header/Footer 用 `shrink-0`，中间主体用 `min-h-0 overflow-y-auto`。列表滚列表区，表单滚主体区，不修改基础 `components/ui/dialog.tsx`。
- 可变长移动端 Sheet 使用 `overflow-y-auto`；自定义 Portal 图片/详情层同时限制 `max-h`、`max-w`，并保留 `px`/`py` 安全边距。

## Java 后端

- 禁止在代码中直接使用全限定类名；先在文件头 import，再使用短类名。
- 所有 Service 使用 `@Cacheable` / `@CacheEvict` 管理缓存；`create`、`update`、`delete` 时清除相关缓存。
- 需要当前用户时统一调用 `SecurityUtils.requireCurrentUserId()`，禁止硬编码 `userId`。

## 开发文档

- `dev-docs/` 顶层只保留 `1-todo/`、`2-in-progress/`、`3-completed/`，禁止散放未分类文档。
- 待办存入 `1-todo/`，文件名为 `YYYY-MM-DD-待办事项.md`。
- 进行中需求存入 `2-in-progress/YYYYMMDD-需求名/`；需求名可使用中文，每个目录固定包含 `development-plan.md`、`progress-tracking.md`、`technical-debt.md`。
- 需求完成后将整个需求目录移入 `3-completed/`，并补充高信息密度的 `completion-summary.md`，记录最终范围、关键实现、验证结果和遗留事项。

## 数据库迁移

- Flyway 迁移统一放在 `ai-fusion-video/src/main/resources/db/migration/`，文件名使用 `V1.<产品主版本>.<产品次版本>.<产品修订版本>.<迁移序号>__<英文描述>.sql`。
- 第一个 `V1` 是为兼容历史数据库永久保留的固定前缀；其后的产品版本三段必须与迁移所属版本一致。例如产品版本 `1.0.0` 使用 `V1.1.0.0.<迁移序号>__...sql`。
- 同一产品版本的迁移序号从 `0` 开始连续递增，描述使用小写 snake_case；创建迁移前先检查目录和目标数据库的 `flyway_schema_history`，避免版本号冲突。
- SQL 中表、字段等数据库 `COMMENT` 使用中文（产品或协议专名除外），禁止英文说明。
- 稳定版本可以增加 `B1.<产品主版本>.<产品次版本>.<产品修订版本>.<迁移序号>__baseline_<产品版本>.sql` 基线；B 的版本必须与快照覆盖的最后一条 V 完全一致。基线必须从执行完全部对应 V 的全新数据库导出，包含最终结构和必要初始化数据，并排除 `flyway_schema_history`、数据库名、`CREATE DATABASE` 与 `USE`。
- 基线只优化全新空数据库的首次部署，不能删除或替代旧 V 迁移；已有历史的数据库仍依赖 V 增量升级。已发布的 B 与 V 一样不可修改或删除。
- 已在任何持久化、共享、测试或生产数据库执行的迁移是不可变历史，禁止修改 SQL 内容、版本或文件名。`V1__init_mysql.sql` 以及 `V1.0.2.0.1` 至 `V1.0.6.1.5` 是历史兼容例外，必须原样保留。
- 尚未发布且只在可丢弃的本地开发库执行过的迁移需要改名时，优先重建本地数据库；必须保留数据时，只能在确认 SQL 内容与 checksum 未变化后同步本地 `flyway_schema_history.version` 和 `script`，随后执行 Flyway validate。禁止用这种方式改写共享环境历史，`flyway repair` 不能把旧版本映射为新版本。
- 未发布迁移若已在本地执行，用户明确要求修改语句时，需同步本地结构和 `flyway_schema_history.checksum` 并完成 validate；禁止改共享环境或增加代码 fallback。
- 新增或调整迁移后，至少在 `ai-fusion-video/` 目录运行 `./mvnw -Dtest=FlywayMigrationNamingTests test`，并通过应用启动或等价方式完成 Flyway validate。详细约定见迁移目录下的 `README.md`。
