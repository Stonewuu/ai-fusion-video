package com.stonewu.fusion.service.ai.run.kernel;

import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;

public interface AgentKernelSnapshotBuilder {

    AgentKernelSnapshot build(AgentKernelSnapshotPayload payload);

    AgentKernelSnapshot build(AgentKernelSpec spec);
}
