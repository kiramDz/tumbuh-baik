"use client";

import { triggerForecastRun } from "@/lib/fetch/files.fetch";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

const RunForecastButton = () => {
  const handleClick = async () => {
    toast.info("Menjalankan forecast...");
    try {
      const result = await triggerForecastRun();
      toast.success("Forecast berhasil dijalankan!");
      console.log("Forecast result:", result);
    } catch (err) {
      toast.error("Gagal menjalankan forecast");
    }
  };

  return (
    <Button onClick={handleClick} variant="outline">
      Jalankan Holt-Winter Sekarang
    </Button>
  );
};

export default RunForecastButton;
