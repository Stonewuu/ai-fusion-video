package com.stonewu.fusion.service.storage;

import com.stonewu.fusion.common.BusinessException;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Registry of built-in S3-compatible provider profiles.
 */
@Component
public class StorageProviderRegistry {

    public static final String GENERIC_S3 = "generic_s3";

    private final Map<String, StorageProviderProfile> profiles;

    public StorageProviderRegistry() {
        List<StorageProviderProfile> values = List.of(
                new StorageProviderProfile(
                        GENERIC_S3,
                        "S3 Compatible",
                        null,
                        null,
                        "us-east-1",
                        true,
                        true,
                        false,
                        true
                ),
                new StorageProviderProfile(
                        "aws_s3",
                        "AWS S3",
                        "https://s3.{region}.amazonaws.com",
                        "https://{bucket}.s3.{region}.amazonaws.com/{key}",
                        "us-east-1",
                        false,
                        false,
                        true,
                        true
                ),
                new StorageProviderProfile(
                        "aliyun_oss",
                        "Alibaba Cloud OSS",
                        "https://s3.oss-{region}.aliyuncs.com",
                        "https://{bucket}.oss-{region}.aliyuncs.com/{key}",
                        null,
                        false,
                        false,
                        false,
                        true
                ),
                new StorageProviderProfile(
                        "tencent_cos",
                        "Tencent Cloud COS",
                        "https://cos.{region}.myqcloud.com",
                        "https://{bucket}.cos.{region}.myqcloud.com/{key}",
                        null,
                        false,
                        false,
                        false,
                        true
                ),
                new StorageProviderProfile(
                        "qiniu_kodo",
                        "Qiniu Kodo",
                        "https://s3.{region}.qiniucs.com",
                        null,
                        null,
                        false,
                        true,
                        false,
                        true
                ),
                new StorageProviderProfile(
                        "ctyun_zos",
                        "CTYun ZOS",
                        null,
                        null,
                        "us-east-1",
                        true,
                        true,
                        false,
                        true
                ),
                new StorageProviderProfile(
                        "minio",
                        "MinIO",
                        null,
                        null,
                        "us-east-1",
                        true,
                        true,
                        false,
                        true
                )
        );

        Map<String, StorageProviderProfile> map = new LinkedHashMap<>();
        for (StorageProviderProfile value : values) {
            map.put(value.provider(), value);
        }
        this.profiles = Map.copyOf(map);
    }

    public StorageProviderProfile get(String provider) {
        StorageProviderProfile profile = profiles.get(provider);
        if (profile == null) {
            throw new BusinessException("不支持的存储厂商: " + provider);
        }
        return profile;
    }

    public StorageProviderProfile getDefault() {
        return profiles.get(GENERIC_S3);
    }

    public List<StorageProviderProfile> list() {
        return List.copyOf(profiles.values());
    }
}
