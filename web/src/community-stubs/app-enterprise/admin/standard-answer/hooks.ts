import type { StandardAnswerCategory } from "@/lib/types";

/** Standard answers are not in the Community Edition. */
export const useStandardAnswerCategories = () => ({
  data: undefined as StandardAnswerCategory[] | undefined,
  isLoading: false,
  error: undefined as unknown,
  refreshStandardAnswerCategories: async () => undefined,
});
