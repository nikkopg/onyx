"use client";

import type { ReactNode } from "react";

/** The default query controller context (chat only) stays in effect. */
export function QueryControllerProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
