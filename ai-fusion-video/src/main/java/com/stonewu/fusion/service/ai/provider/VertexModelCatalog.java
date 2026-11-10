package com.stonewu.fusion.service.ai.provider;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.service.ai.google.GoogleModelFamily;
import okhttp3.HttpUrl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;

/** 读取 Google 发布者目录，分别标记模型类型与当前渠道的导入能力。 */
final class VertexModelCatalog {
    static final String ENDPOINT = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models";

    List<RemoteModelVO> load(String endpoint, Function<String, String> request) {
        var models = new LinkedHashMap<String, RemoteModelVO>();
        var seenTokens = new HashSet<String>();
        String pageToken = null;
        do {
            var url = HttpUrl.get(endpoint).newBuilder().addQueryParameter("pageSize", "100")
                    .addQueryParameter("listAllVersions", "true");
            if (pageToken != null) {
                url.addQueryParameter("pageToken", pageToken);
            }
            JSONObject response = JSONUtil.parseObj(request.apply(url.build().toString()));
            var entries = response.getJSONArray("publisherModels");
            if (entries == null) {
                throw new BusinessException("Google 模型目录响应缺少 publisherModels，未导入模型");
            }
            for (JSONObject entry : entries.toList(JSONObject.class)) {
                String name = entry.getStr("name", "");
                String prefix = "publishers/google/models/";
                if (!name.startsWith(prefix) || name.length() == prefix.length()) {
                    throw new BusinessException("Google 模型目录返回了无效的模型标识");
                }
                String code = name.substring(prefix.length());
                Integer type = GoogleModelFamily.type(code);
                if (type == null) {
                    continue;
                }
                String display = entry.getStr("displayName");
                models.putIfAbsent(code, RemoteModelVO.builder().id(code)
                        .displayName(display == null || display.isBlank() ? code : display)
                        .ownedBy("google").providerPlatform("vertex_ai").modelType(type)
                        .modelProtocol("vertex_ai").inferredMetadata(true).build());
            }
            pageToken = response.getStr("nextPageToken");
            if (pageToken != null && pageToken.isBlank()) {
                pageToken = null;
            }
            if (pageToken != null && !seenTokens.add(pageToken)) {
                throw new BusinessException("Google 模型目录返回重复分页标识，未返回不完整列表");
            }
        } while (pageToken != null);
        return new ArrayList<>(models.values());
    }

}
