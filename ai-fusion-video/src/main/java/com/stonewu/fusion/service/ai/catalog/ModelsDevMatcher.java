package com.stonewu.fusion.service.ai.catalog;

import com.stonewu.fusion.service.ai.catalog.ModelsDevCatalogService.Entry;
import com.stonewu.fusion.service.ai.catalog.ModelsDevCatalogService.Snapshot;

import java.util.Comparator;
import java.util.List;
import java.util.Arrays;
import java.util.Locale;

/** 按完整标识优先排序，名称、分词及小范围拼写差异用于用户手动搜索。 */
public final class ModelsDevMatcher {
    private ModelsDevMatcher() {
    }

    public static SearchResult search(Snapshot catalog, String query, int modelType, String providerId) {
        String term = normalize(query);
        var terms = Arrays.stream(query.split("\\s+")).map(ModelsDevMatcher::normalize).filter(value -> !value.isEmpty()).toList();
        var ranked = catalog.models().stream().filter(model -> model.modelType() == modelType)
                .map(model -> new Ranked(model, score(model, term, terms))).filter(value -> value.score() > 0)
                .sorted(Comparator.comparingInt(Ranked::score).reversed()
                        .thenComparing(value -> value.model().modelId())
                        .thenComparing(value -> value.model().providerId())).toList();
        var providers = ranked.stream().map(value -> new Provider(value.model().providerId(), value.model().providerName()))
                .distinct().sorted(Comparator.comparing(Provider::name)).toList();
        var filtered = ranked.stream().filter(value -> providerId == null || providerId.equals(value.model().providerId())).toList();
        return new SearchResult(catalog.fetchedAt(), filtered.size(), filtered.stream().limit(60).map(Ranked::model).toList(), providers);
    }

    private static int score(Entry model, String query, List<String> terms) {
        String id = normalize(model.modelId());
        String name = normalize(model.name());
        String canonical = normalize(model.canonicalModelId());
        int firstParty = model.canonicalModelId() != null
                && model.canonicalModelId().startsWith(model.providerId() + "/") ? 5 : 0;
        if (query.isEmpty()) {
            return 10 + firstParty;
        }
        if (id.equals(query) || name.equals(query) || canonical.equals(query)) {
            return 100 + firstParty;
        }
        if (id.contains(query) || name.contains(query) || canonical.contains(query)) {
            return 80 + firstParty;
        }
        String corpus = id + name + canonical + normalize(model.providerName());
        if (terms.size() > 1 && terms.stream().allMatch(corpus::contains)) {
            return 60 + firstParty;
        }
        int edits = distance(query, id);
        if (query.length() >= 4 && edits <= Math.min(3, query.length() / 4)) {
            return 40 + firstParty - edits * 8;
        }
        return corpus.contains(query) ? 20 + firstParty : 0;
    }

    private static String normalize(String value) {
        return value == null ? "" : value.toLowerCase(Locale.ROOT).replaceAll("[^\\p{L}\\p{N}]", "");
    }

    private static int distance(String left, String right) {
        if (Math.abs(left.length() - right.length()) > 3) {
            return 4;
        }
        int[] previous = new int[right.length() + 1];
        for (int index = 0; index < previous.length; index++) {
            previous[index] = index;
        }
        for (int row = 1; row <= left.length(); row++) {
            int[] current = new int[right.length() + 1];
            current[0] = row;
            for (int column = 1; column <= right.length(); column++) {
                int replacement = previous[column - 1] + (left.charAt(row - 1) == right.charAt(column - 1) ? 0 : 1);
                current[column] = Math.min(replacement, Math.min(previous[column] + 1, current[column - 1] + 1));
            }
            previous = current;
        }
        return previous[right.length()];
    }

    private record Ranked(Entry model, int score) {
    }

    public record Provider(String id, String name) {
    }

    public record SearchResult(String fetchedAt, int total, List<Entry> models, List<Provider> providers) {
    }
}
