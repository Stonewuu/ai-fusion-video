# 技术债务

未新增代码 fallback 或技术债务。

## 历史数据说明

修复从应用重启后的新写入开始生效。旧版本在 UTC 运行环境中写入的无时区 `DATETIME` 无法与其他时区写入的数据可靠区分，因此未执行可能误改数据的批量时间迁移；旧记录再次更新后，其 `update_time` 会按统一时区写入。

部署覆盖默认时区时，`APP_TIME_ZONE` 应使用 IANA Zone ID，`MYSQL_SESSION_TIME_ZONE` 应设置为对应的 MySQL UTC 偏移；默认分别为 `Asia/Shanghai` 和 `+08:00`。
