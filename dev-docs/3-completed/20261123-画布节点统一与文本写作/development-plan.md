# 画布节点统一与文本 AI 写作

在快捷生成的基础上统一画布节点外观，并让文本节点可以直接接入生成能力，由 AI 写入正文。

1. 节点统一：除分组外，节点均采用框外标题、内容框、两侧连接点和右侧“添加后续”；媒体铺满内容框，文本使用易读字号；双击标题改名。
2. 面板跟随：生成表单与选中工具条渲染在画布视口层，随节点平移、反向缩放保持原尺寸；拖动节点不打开表单。
3. 文本面板：点击文本节点打开正文面板，失焦保存；“用这段文字”一键新建已连线的生图、生视频或生成文本节点。
4. 生成文本接入快捷生成：后端输入预览接受生成文本节点；表单选择文本模型、填写写作要求并生成。
5. 结果写回：生成文本的当前结果可写回为其提供写作素材的文本节点正文，写回后结果转为待更新。

文本生成沿用画布运行的异步任务（排队、重试、审核与鉴权），不新增绕开运行体系的一次性写入接口。

## 画布专用组件

页面按钮与输入框（36px、描边、大圆角）放在画布浮层和节点上显得笨重，与点阵背景和节点卡片不协调。为画布区域建立独立的紧凑组件，与 `components/ui/` 并存，互不修改。

### 范围

- 适用：节点、节点快捷面板（生成表单、正文面板）、悬浮工具条、选中工具条、右键与添加菜单、底部状态栏、空状态。
- 不适用：属性侧栏、Dialog、Sheet 及画布以外的页面，继续使用 `components/ui/`。

### 设计

- 视觉方向：紧凑填充式。控件默认 32px，节点内 28px / 24px；32px 用 `rounded-md`，更小用 `rounded-sm`。
- 输入：浅底无描边，悬停显出细边框，聚焦时品牌色描边与 2px 外圈。
- 按钮语义：`primary`、`secondary`、`ghost`、`danger`、`overlay`，以及生成主操作 `image` / `video` / `text`。
- 图标按钮统一带悬停提示与快捷键。

### 组件（`components/canvas/ui/`）

- `canvas-button.tsx`：`CanvasButton`、`canvasButtonVariants`。
- `canvas-icon-button.tsx`：`CanvasIconButton`（label 即无障碍名称与提示）。
- `canvas-field.tsx`：`CanvasInput`、`CanvasTextarea`。
- `canvas-picker.tsx`：`CanvasPicker`（下拉触发器为画布外观，列表沿用共享样式）。
- `canvas-segmented.tsx`：`CanvasSegmented`。
- `canvas-surface.tsx`：`CanvasFloatingBar`、`CanvasBarDivider`、`CanvasPanel`。

协作规则已写入 `AGENTS.md`「画布组件」。
