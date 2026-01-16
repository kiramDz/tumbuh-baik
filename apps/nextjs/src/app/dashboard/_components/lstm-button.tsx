"use client"

import { triggerLSTMForecast } from "@/lib/fetch/files.fetch"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"
import { useQueryClient } from "@tanstack/react-query"

const RunLSTMButton = () => {
    const queryClient = useQueryClient();
    const handleClick = async () => {
        toast.info("Menjalankan forecast LSTM...");
        try {
            const result = await triggerLSTMForecast();
            toast.success("Forecast LSTM berhasil dijalankan!");

            queryClient.invalidateQueries({ queryKey: ["lstm-daily"] });

            console.log("Forecast LSTM result:", result);
        } catch (err: any) {
            toast.error(`Gagal menjalankan forecast LSTM: ${err?.message || "Unknown error"}`);
        }
    };

    return (
        <Button onClick={handleClick} variant="outline" className="text-base">
            Jalankan Forecast LSTM
        </Button>
    );
}

export default RunLSTMButton;