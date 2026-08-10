package com.stonewu.fusion.service.storage;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import software.amazon.awssdk.auth.credentials.AwsBasicCredentials;
import software.amazon.awssdk.auth.credentials.StaticCredentialsProvider;
import software.amazon.awssdk.core.checksums.RequestChecksumCalculation;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.S3ClientBuilder;
import software.amazon.awssdk.services.s3.S3Configuration;

import jakarta.annotation.PreDestroy;
import java.net.URI;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Caches thread-safe AWS SDK S3 clients by effective storage configuration.
 */
@Component
@Slf4j
public class S3ClientFactory {

    private final Map<ClientCacheKey, S3Client> clients = new ConcurrentHashMap<>();

    public S3Client getClient(ResolvedS3StorageConfig config) {
        ClientCacheKey key = ClientCacheKey.of(config);
        return clients.computeIfAbsent(key, ignored -> buildClient(config));
    }

    public void invalidate(Long configId) {
        if (configId == null) {
            return;
        }
        for (Map.Entry<ClientCacheKey, S3Client> entry : clients.entrySet()) {
            if (Objects.equals(entry.getKey().configId(), configId)
                    && clients.remove(entry.getKey(), entry.getValue())) {
                closeQuietly(entry.getValue());
            }
        }
    }

    @PreDestroy
    public void closeAll() {
        clients.values().forEach(this::closeQuietly);
        clients.clear();
    }

    private S3Client buildClient(ResolvedS3StorageConfig config) {
        AwsBasicCredentials credentials = AwsBasicCredentials.create(config.accessKey(), config.secretKey());

        S3ClientBuilder builder = S3Client.builder()
                .region(Region.of(config.signingRegion()))
                .credentialsProvider(StaticCredentialsProvider.create(credentials))
                .requestChecksumCalculation(resolveChecksumCalculation(config.checksumCalculation()))
                .serviceConfiguration(S3Configuration.builder()
                        .pathStyleAccessEnabled(config.pathStyleAccessEnabled())
                        .chunkedEncodingEnabled(config.chunkedEncodingEnabled())
                        .build());

        if (config.endpoint() != null) {
            builder.endpointOverride(URI.create(config.endpoint()));
        }

        log.info("[S3ClientFactory] 创建 S3 client: provider={}, endpoint={}, region={}, pathStyle={}, chunked={}",
                config.provider(), config.endpoint(), config.signingRegion(),
                config.pathStyleAccessEnabled(), config.chunkedEncodingEnabled());
        return builder.build();
    }

    private RequestChecksumCalculation resolveChecksumCalculation(String value) {
        if (value == null) {
            return RequestChecksumCalculation.WHEN_REQUIRED;
        }
        try {
            return RequestChecksumCalculation.valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException ignored) {
            return RequestChecksumCalculation.WHEN_REQUIRED;
        }
    }

    private void closeQuietly(S3Client client) {
        if (client == null) {
            return;
        }
        try {
            client.close();
        } catch (Exception e) {
            log.warn("[S3ClientFactory] 关闭 S3 client 失败: {}", e.getMessage());
        }
    }

    record ClientCacheKey(
            Long configId,
            String provider,
            String endpoint,
            String signingRegion,
            String accessKey,
            int secretHash,
            boolean pathStyleAccessEnabled,
            boolean chunkedEncodingEnabled,
            String checksumCalculation
    ) {
        static ClientCacheKey of(ResolvedS3StorageConfig config) {
            Long configId = config.source() != null ? config.source().getId() : null;
            return new ClientCacheKey(
                    configId,
                    config.provider(),
                    config.endpoint(),
                    config.signingRegion(),
                    config.accessKey(),
                    Objects.hashCode(config.secretKey()),
                    config.pathStyleAccessEnabled(),
                    config.chunkedEncodingEnabled(),
                    config.checksumCalculation()
            );
        }
    }
}
