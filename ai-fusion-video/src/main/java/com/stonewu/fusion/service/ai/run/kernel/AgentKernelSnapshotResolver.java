package com.stonewu.fusion.service.ai.run.kernel;

import java.util.List;

public interface AgentKernelSnapshotResolver {

    AgentKernelSnapshot resolve(
            String persistedCanonicalJson,
            String persistedFingerprint,
            long currentModelConfigVersion,
            List<ToolManifestSnapshot> currentTools);
}
