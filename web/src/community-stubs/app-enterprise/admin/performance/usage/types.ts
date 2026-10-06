export interface UsageReport {
  report_name: string;
  requestor: string | null;
  time_created: string;
  period_from: string | null;
  period_to: string | null;
}
