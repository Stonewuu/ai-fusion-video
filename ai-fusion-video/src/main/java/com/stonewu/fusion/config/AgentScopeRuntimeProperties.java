package com.stonewu.fusion.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Bounded execution settings for the AgentScope runtime.
 */
@ConfigurationProperties(prefix = "app.agentscope.runtime")
public class AgentScopeRuntimeProperties {

    private int stateThreads = defaultStateThreads(Runtime.getRuntime().availableProcessors());
    private int journalThreads = defaultJournalThreads(Runtime.getRuntime().availableProcessors());
    private int modelThreads = defaultModelThreads(Runtime.getRuntime().availableProcessors());
    private int toolThreads = defaultToolThreads(Runtime.getRuntime().availableProcessors());

    public static int defaultStateThreads(int processors) {
        return stateOrJournalThreads(processors);
    }

    public static int defaultJournalThreads(int processors) {
        return stateOrJournalThreads(processors);
    }

    public static int defaultModelThreads(int processors) {
        return modelOrToolThreads(processors);
    }

    public static int defaultToolThreads(int processors) {
        return modelOrToolThreads(processors);
    }

    public int getStateThreads() {
        return stateThreads;
    }

    public void setStateThreads(int stateThreads) {
        this.stateThreads = positive("stateThreads", stateThreads);
    }

    public int getJournalThreads() {
        return journalThreads;
    }

    public void setJournalThreads(int journalThreads) {
        this.journalThreads = positive("journalThreads", journalThreads);
    }

    public int getModelThreads() {
        return modelThreads;
    }

    public void setModelThreads(int modelThreads) {
        this.modelThreads = positive("modelThreads", modelThreads);
    }

    public int getToolThreads() {
        return toolThreads;
    }

    public void setToolThreads(int toolThreads) {
        this.toolThreads = positive("toolThreads", toolThreads);
    }

    private static int stateOrJournalThreads(int processors) {
        return (int) Math.max(4L, Math.min(32L, (long) positive("processors", processors) * 2L));
    }

    private static int modelOrToolThreads(int processors) {
        return (int) Math.max(8L, Math.min(64L, (long) positive("processors", processors) * 4L));
    }

    private static int positive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be greater than zero");
        }
        return value;
    }
}
