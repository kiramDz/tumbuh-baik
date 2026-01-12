"use client";

import { triggerForecastRun } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { useQueryClient } from "@tanstack/react-query";

const RunForecastButton = () => {
  const queryClient = useQueryClient();
  const handleClick = async () => {
    toast.info("Menjalankan forecast...");
    try {
      const result = await triggerForecastRun();
      toast.success("Forecast berhasil dijalankan!");

      queryClient.invalidateQueries({ queryKey: ["hw-daily"] });

      console.log("Forecast result:", result);
    } catch (err: any) {
      toast.error(`Gagal menjalankan forecast: ${err?.message || "Unknown error"}`);
    }
  };

  return (
    <Button onClick={handleClick} variant="outline">
      Jalankan Holt-Winter
    </Button>
  );
};

export default RunForecastButton;
