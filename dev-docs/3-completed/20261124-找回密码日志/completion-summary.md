# 找回密码日志修复完成

Docker 部署的日志级别为 `ERROR`，而管理员找回密码验证码通过 `AuthController` 的 `WARN` 日志输出，导致页面提示成功后容器日志仍看不到验证码。

已在 Docker 配置中为 `AuthController` 单独设置 `WARN`。线上 v1.4.1 通过同一启动参数生效，源码默认配置也已修复。

扩展了现有 `LoggingProfilesTests`，启动日志系统验证常规 Docker 日志过滤和认证控制器警告输出，两项用例通过。服务器本机应用入口申请验证码返回成功，容器日志已显示密码找回提示与 64 位验证码。

重新申请验证码后，可使用 `docker logs --tail 100 -f fusion-backend` 查看；使用容器名可以避免容器更新后继续查看旧容器编号。
