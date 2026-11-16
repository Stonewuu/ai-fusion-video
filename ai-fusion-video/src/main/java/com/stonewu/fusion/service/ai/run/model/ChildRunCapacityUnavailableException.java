package com.stonewu.fusion.service.ai.run.model;

/** 内部调度信号：父任务的子任务名额已满，调用方应在事务结束后等待再申请。 */
public final class ChildRunCapacityUnavailableException extends RuntimeException {
    public ChildRunCapacityUnavailableException() {
        super("父任务的子任务名额已满", null, false, false);
    }
}
