"use client";

import type { MinimalOnyxDocument } from "@/lib/search/types";

export interface SearchResultsProps {
  onDocumentClick: (doc: MinimalOnyxDocument) => void;
}

export default function SearchUI(_props: SearchResultsProps) {
  return null;
}
