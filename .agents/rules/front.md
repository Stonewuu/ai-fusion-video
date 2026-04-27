---
trigger: glob
globs: *.js, *.tsx
---

# 前端开发经验规则

## 1. `overflow-hidden` 容器内的弹出层处理

**场景**：当组件使用 `overflow-hidden`（如 glow-menu 的 `<nav>`）来裁切视觉效果（光晕、阴影扩散等），但内部又需要弹出下拉菜单、Tooltip 等溢出内容时。

**问题**：直接在 `overflow-hidden` 容器内渲染弹出层会被裁切不可见。

**解决方案**：使用 **React Portal**（`createPortal`）将弹出层渲染到 `document.body`，配合 `useRef` + `getBoundingClientRect()` 精确定位到触发按钮下方。

```tsx
// 1. 触发按钮用 ref 获取位置
const btnRef = useRef<HTMLButtonElement>(null);
const [pos, setPos] = useState({ top: 0, right: 0 });

useEffect(() => {
  if (open && btnRef.current) {
    const rect = btnRef.current.getBoundingClientRect();
    setPos({ top: rect.bottom + 8, right: window.innerWidth - rect.right });
  }
}, [open]);

// 2. 按钮放在 overflow-hidden 容器内
<button ref={btnRef} onClick={() => setOpen(!open)}>
  触发
</button>;

// 3. 弹出层通过 Portal 渲染到 body
{
  createPortal(
    <div className="fixed z-[61]" style={{ top: pos.top, right: pos.right }}>
      弹出内容
    </div>,
    document.body,
  );
}
```

**适用组件**：glow-menu 内的下拉菜单、带 `overflow-hidden` 的卡片内的 Tooltip/Popover 等。

## 2. Dialog / 弹窗小视口兜底规则

**场景**：表单型 Dialog、长列表 Dialog、自定义 Portal 详情弹层在笔记本小窗口、浏览器缩放、移动端横屏时，内容高度可能超过视口。

**问题**：如果只给弹窗设宽度，不限制高度或不拆分内部滚动区，底部按钮会被挤出视口，用户无法提交或关闭。

**解决方案**：不要改基础 `components/ui/dialog.tsx`，在具体使用处套统一结构。

```tsx
<DialogContent className="sm:max-w-lg max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden">
  <DialogHeader className="shrink-0" />

  <div className="min-h-0 overflow-y-auto px-1 -mx-1">
    {/* 长表单 / 长列表内容 */}
  </div>

  <DialogFooter className="shrink-0" />
</DialogContent>
```

**补充规则**：

1. `DialogContent` 至少加 `max-h-[calc(100vh-2rem)] flex flex-col overflow-hidden`
2. `DialogHeader`、`DialogFooter` 加 `shrink-0`，保证头尾不被压缩
3. 中间内容区单独加 `min-h-0 overflow-y-auto`
4. 列表型弹窗优先让列表区自己滚动；表单型弹窗优先让表单主体滚动
5. 移动端右侧详情 `Sheet` 如果内容可能变长，使用 `overflow-y-auto`，不要写死 `overflow-hidden`
6. 自定义 Portal 图片/详情弹层要同时限制 `max-h` / `max-w`，并给外层留 `px`、`py` 安全边距

# 布局高度管理规则

## 整体布局结构

Dashboard 布局采用 `h-screen overflow-hidden flex flex-col` 限制整个页面不超出视口：

```
<div h-screen overflow-hidden flex flex-col>       ← 视口高度限制
  <header fixed>                                    ← fixed 定位，不占文档流
  <motion.div flex-1 min-h-0 pt-20>               ← 填满剩余空间，pt-20 为 fixed header 留位
    <sidebar fixed>                                 ← fixed 定位
    <main flex-1 min-h-0 overflow-auto>            ← 内容区，内部滚动
      <div max-w-7xl mx-auto h-full>               ← 居中容器，传递高度
        {children}
      </div>
    </main>
  </motion.div>
</div>
```

## 页面高度使用规则

1. **普通页面**（概览、设置等）：不需要特殊处理，内容超出时 `<main>` 自动滚动
2. **全高度页面**（剧本编辑器等三栏布局）：使用 `h-full` 继承父级高度，**禁止使用 `h-[calc(100vh-Xrem)]` 硬编码**
3. 需要内部独立滚动的子区域（如左栏、中栏、右栏），自行添加 `overflow-auto`

## 关键 class 说明

- `min-h-0`：flex 子项必须加，否则无法缩小到比内容更小的尺寸
- `h-full`：继承父级高度，需要确保父级链上每层都有明确的高度
- `overflow-auto`：允许内部滚动，配合高度限制使用

# shadcn 组件管理规则

## 安装组件

使用 **pnpm** 安装 shadcn 组件，在 `ai-fusion-video-web` 目录下执行：

```bash
pnpm dlx shadcn@latest add <component-name> --yes
```

**示例**：
```bash
pnpm dlx shadcn@latest add select --yes
pnpm dlx shadcn@latest add dialog --yes
```

## 注意事项

1. **禁止使用 npx**，统一使用 `pnpm dlx`
2. 安装命令必须在 `ai-fusion-video-web` 目录下执行
3. 安装后组件会自动生成到 `components/ui/` 目录
4. 使用时通过 `@/components/ui/<component>` 路径导入
5. **严禁修改 `components/ui/` 下的组件源码**，样式调整通过使用处的 `className` 覆盖

## Select 组件使用规范（base-ui 风格）

项目 Select 基于 `@base-ui/react`（style: base-maia），**严禁修改 `components/ui/select.tsx` 源码**，样式通过使用处 `className` 覆盖。

1. **`items` prop 必传**：`<Select items={options}>` 中 `options` 为 `{ value, label }[]`，不传则 trigger 显示原始 value 而非 label，且无法自适应宽度
2. **`SelectValue` 用 `placeholder` 属性**：`<SelectValue placeholder="选择" />`，禁止传 children
3. **`SelectGroup` 包裹 `SelectItem`**：`SelectContent > SelectGroup > SelectItem`，提供选项与弹窗边缘的间距
4. **字号通过 `className` 覆盖**：`SelectTrigger`、`SelectContent`、`SelectItem` 三处同时传 `className="text-xs"` 缩小字号（组件默认 `text-sm`）
5. **禁止 `rounded-lg!` 等强制覆盖**：保持组件默认的 `rounded-4xl` 药丸造型

