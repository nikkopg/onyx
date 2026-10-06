import type { BaseFilters, SearchFullResponse } from "@/lib/search/types";

/** The search UI backend is not in the Community Edition: no documents match. */
export async function searchDocuments(
  _query: string,
  _options?: {
    filters?: BaseFilters;
    numHits?: number;
    includeContent?: boolean;
    signal?: AbortSignal;
  }
): Promise<SearchFullResponse> {
  return { all_executed_queries: [], search_docs: [] };
}
