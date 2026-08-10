package com.stonewu.fusion.service.storage;

/**
 * Defaults and URL templates for an S3-compatible object storage provider.
 */
public record StorageProviderProfile(
        String provider,
        String label,
        String endpointTemplate,
        String publicUrlTemplate,
        String defaultRegion,
        boolean endpointRequired,
        boolean pathStyleAccessEnabled,
        boolean chunkedEncodingEnabled,
        boolean publicUrlInferable
) {
}
