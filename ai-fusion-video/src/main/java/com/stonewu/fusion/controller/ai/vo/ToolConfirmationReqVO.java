package com.stonewu.fusion.controller.ai.vo;

import lombok.Data;

import java.util.List;

/** User decisions for one durable AgentScope permission pause. */
@Data
public class ToolConfirmationReqVO {

    private String runId;
    private String replyId;
    private List<DecisionVO> decisions;

    @Data
    public static class DecisionVO {
        private String toolCallId;
        private Boolean approved;
    }
}
