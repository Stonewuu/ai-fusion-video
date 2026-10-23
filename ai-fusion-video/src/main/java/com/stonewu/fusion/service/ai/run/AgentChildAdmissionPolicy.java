package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.ai.run.model.StartChildAgentRunCommand;

public interface AgentChildAdmissionPolicy {
    void admit(AgentRun parent, StartChildAgentRunCommand command);
}
