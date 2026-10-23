package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.service.resource.model.ResourceCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceMutation;

/**
 * 跟随业务命令的同一事务执行；校验或引用登记失败必须回滚正文和审计。
 */
public interface ResourceMutationObserver {
    default void before(ResourceCommand command, ResourceCommandContext context) {
    }

    void validateAndApply(ResourceMutation mutation, ResourceCommandContext context);
}
