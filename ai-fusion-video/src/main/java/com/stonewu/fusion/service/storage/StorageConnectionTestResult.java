package com.stonewu.fusion.service.storage;

/**
 * Result of a storage connectivity test.
 */
public record StorageConnectionTestResult(
        boolean success,
        String message,
        String publicUrl
) {
    public static StorageConnectionTestResult success(String message, String publicUrl) {
        return new StorageConnectionTestResult(true, message, publicUrl);
    }

    public static StorageConnectionTestResult failure(String message, String publicUrl) {
        return new StorageConnectionTestResult(false, message, publicUrl);
    }
}
