SET @initial_admin_user_id := (
    SELECT su.id
    FROM sys_user su
    INNER JOIN sys_user_role sur ON sur.user_id = su.id
    INNER JOIN sys_role sr ON sr.id = sur.role_id
    WHERE sr.code = 'admin'
    ORDER BY su.id
    LIMIT 1
);

INSERT INTO afv_team (name, description, owner_user_id, status, deleted, create_time, update_time)
SELECT CONCAT(su.username, '的团队'), '开源版默认团队', su.id, 1, 0, NOW(), NOW()
FROM sys_user su
WHERE su.id = @initial_admin_user_id
  AND @initial_admin_user_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM afv_team
  );

SET @single_team_id := (
    SELECT id
    FROM afv_team
    ORDER BY id
    LIMIT 1
);

INSERT INTO afv_team_member (team_id, user_id, role, status, join_time, deleted, create_time, update_time)
SELECT @single_team_id,
       su.id,
       CASE WHEN su.id = @initial_admin_user_id THEN 1 ELSE 3 END,
       1,
       NOW(),
       0,
       NOW(),
       NOW()
FROM sys_user su
WHERE @single_team_id IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM afv_team_member tm
      WHERE tm.team_id = @single_team_id
        AND tm.user_id = su.id
  );

INSERT INTO afv_system_config (config_key, config_value, remark, deleted, create_time, update_time)
SELECT 'allow_register', 'false', '是否允许公开注册', 0, NOW(), NOW()
WHERE NOT EXISTS (
    SELECT 1
    FROM afv_system_config
    WHERE config_key = 'allow_register'
);