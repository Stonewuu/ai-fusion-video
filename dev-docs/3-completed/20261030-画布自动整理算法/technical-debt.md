# 维护边界

本次改进画布工具栏的自动整理。服务端助手工具的布局仍是独立实现，没有浏览器测量数据；本次不修改助手协议。布局引擎不承诺复杂图的数学全局最优。

展开分组作为整体保留内部布局。分组内局部整理受现有边界约束，空间不足时明确提示调整边界，不移动未选中的父分组。锁定子节点所在的整个分组不能通过整体移动改变该节点的位置。

一次命令最多移动 500 个节点，沿用服务端命令数量约束。单次 Worker 计算超过 15 秒会被终止并提示；单个路由网格限制为 75 万个交点，最多同时缓存两组网格，避免复杂图无上限占用内存。该上限是资源约束，不代表已经验证 500 节点的交互性能。

路径由节点几何数据重新计算，不写入业务图协议。拖动时使用跟随端点的曲线预览，停止后更新避让路径；路径计算失败会显示原因。浏览器帧率和图片、视频解码开销尚未测量，离线算法耗时不能替代本机交互验收。

配置依据：[ELK Layered](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html)、[端口约束](https://eclipse.dev/elk/reference/options/org-eclipse-elk-portConstraints.html)、[交叉优化参数](https://eclipse.dev/elk/reference/options/org-eclipse-elk-layered-crossingMinimization-greedySwitch-type.html)。最终节点位置仍由应用层执行锁定和边界约束，不把 ELK 的节点优先级当作固定坐标保证。
