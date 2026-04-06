import { PreprocessingReport } from "@/lib/fetch/files.fetch";
import {
  Card,
  CardContent,
  CardHeader,
  CardDescription,
  CardTitle,
} from "@/components/ui/card";
import { Icons } from "@/app/dashboard/_components/icons";
import { Badge } from "@/components/ui/badge";

interface PreprocessingSummaryProps {
  report: PreprocessingReport;
}

export function PreprocessingSummary({ report }: PreprocessingSummaryProps) {
  const summary = report.preprocessing_summary || {};
  const isNasa = report.dataset_type === "nasa";

  // --- BMKG Metrics ---
  const imputationSummary =
    summary.missing_data?.imputation_summary?.per_parameter || {};
  const bmkgOutliers = summary.outliers || {};

  // --- NASA Metrics ---
  const nasaSmoothing = summary.smoothing?.decisions?.reasons || {};
  const nasaOutlierParams = summary.outliers?.by_parameter || {};
  const nasaOutlierTreatment = summary.outliers?.treatment || "none";

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {/* Imputation & data filling Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icons.database className="h-5 w-5" />
            Data Handling & Imputation
          </CardTitle>
          <CardDescription>
            Rules applied to handle missing values
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!isNasa ? (
            <div className="space-y-4">
              <div className="text-sm font-medium">
                Total Imputed:{" "}
                {summary.missing_data?.imputation_summary?.total_imputed || 0}
              </div>
              <div className="grid gap-2 border rounded-md p-4 bg-muted/20">
                {Object.entries(imputationSummary).map(
                  ([param, data]: [string, any]) => (
                    <div
                      key={param}
                      className="flex justify-between items-center text-sm border-b pb-2 last:border-0 last:pb-0"
                    >
                      <span className="font-semibold">{param}</span>
                      <div className="flex items-center gap-4 text-right">
                        <span className="text-muted-foreground w-[40px]">
                          {data.imputed}x
                        </span>
                        <span
                          className="text-muted-foreground hidden sm:inline-block w-[180px] truncate"
                          title={data.method}
                        >
                          {data.method}
                        </span>
                        <Badge
                          variant={
                            data.success_rate >= 90 ? "default" : "secondary"
                          }
                        >
                          {data.success_rate}%
                        </Badge>
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground italic h-full flex items-center justify-center p-4">
              NASA dataset does not require complex spatial imputation.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Outliers & Smoothing Cards */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Icons.activity className="h-5 w-5" />
            Outliers & Variance Control
          </CardTitle>
          <CardDescription>
            How extreme values and noise were treated
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isNasa ? (
            <div className="space-y-4">
              <div className="text-sm font-medium">
                Total Outliers Treated: {summary.outliers?.total_outliers || 0}{" "}
                ({nasaOutlierTreatment})
              </div>
              <div className="grid gap-2 border rounded-md p-4 bg-muted/20 max-h-[300px] overflow-y-auto">
                <div className="text-xs font-bold text-muted-foreground mb-1 uppercase">
                  Smoothing Methods Applied
                </div>
                {Object.entries(nasaSmoothing).map(
                  ([param, reason]: [string, any]) => (
                    <div
                      key={param}
                      className="flex flex-col gap-1 text-sm border-b pb-2 last:border-0 last:pb-0"
                    >
                      <div className="flex justify-between">
                        <span className="font-semibold">{param}</span>
                        <Badge variant="outline">
                          {nasaOutlierParams[param] || 0} Outliers
                        </Badge>
                      </div>
                      <span className="text-xs text-muted-foreground leading-relaxed">
                        {reason}
                      </span>
                    </div>
                  ),
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="text-sm font-medium">Outliers Interpolated</div>
              <div className="grid gap-2 border rounded-md p-4 bg-muted/20">
                {Object.entries(bmkgOutliers)
                  .filter(([_, data]: [string, any]) => data?.count > 0)
                  .map(([param, data]: [string, any]) => (
                    <div
                      key={param}
                      className="flex justify-between items-center text-sm border-b pb-2 last:border-0 last:pb-0"
                    >
                      <span className="font-semibold">{param}</span>
                      <div className="flex gap-4">
                        <span>{data.count} found</span>
                        <Badge variant="secondary">{data.percentage}%</Badge>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
