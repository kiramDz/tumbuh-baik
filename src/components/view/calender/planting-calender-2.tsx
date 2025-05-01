import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { YearlyOption } from "./year-option";
import { Badge } from "@/components/ui/badge";

const items = [
  {
    feature: "Rice",
    musim1: [
      { month: "Jan", status: "tanam", note: "Musim hujan" },
      { month: "Feb", status: "tanam", note: "Musim hujan" },
      { month: "Mar", status: "istirahat", note: "Paska tanam" },
      { month: "Apr", status: "panen", note: "Panen" },
    ],
    musim2: [
      { month: "May", status: "istirahat", note: "" },
      { month: "Jun", status: "istirahat", note: "" },
      { month: "Jul", status: "tanam", note: "Musim tanam ke-2" },
      { month: "Aug", status: "tanam", note: "Musim tanam ke-2" },
      { month: "Sep", status: "panen", note: "Panen" },
      { month: "Oct", status: "panen", note: "Panen" },
      { month: "Nov", status: "panen", note: "" },
      { month: "Dec", status: "panen", note: "" },
    ],
  },
];

function YearlyCalender() {
  return (
    <div className="spaye-y-4">
      <div className="flex items-center w-full justify-between">
        <div className="flex gap-2">
          <Badge className="rounded-md bg-green-300">Tanam</Badge>
          <Badge className="rounded-md bg-yellow-500">Panen</Badge>
          <Badge variant="destructive" className="rounded-md">
            Rehat
          </Badge>
        </div>
        <YearlyOption />
      </div>
      <Table className="bg-background mt-6">
        <TableHeader>
          <TableRow className="border-y-0 *:border-border hover:bg-transparent [&>:not(:last-child)]:border-r">
            <TableCell></TableCell>
            <TableHead className="border-b border-border text-center" colSpan={5}>
              <span>Musim Tanam 1</span>
            </TableHead>
            <TableHead className="border-b border-border text-center" colSpan={5}>
              <span>Musim Tanam 2</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableHeader>
          <TableRow className="*:border-border hover:bg-transparent [&>:not(:last-child)]:border-r">
            <TableCell></TableCell>
            {items[0].musim1.map((calender) => (
              <TableHead key={calender.month} className="h-auto rotate-180 py-3 text-foreground [writing-mode:vertical-lr]">
                {calender.month}
              </TableHead>
            ))}
            {items[0].musim2.map((calender) => (
              <TableHead key={calender.month} className="h-auto rotate-180 py-3 text-foreground [writing-mode:vertical-lr]">
                {calender.month}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {items.map((item) => (
            <TableRow key={item.feature} className="*:border-border [&>:not(:last-child)]:border-r">
              <TableHead className="font-medium text-foreground">{item.feature}</TableHead>
              {[...item.musim1, ...item.musim2].map((calender, index) => {
                const statusColorMap: Record<"tanam" | "panen" | "istirahat", string> = {
                  tanam: "emerald-500",
                  panen: "yellow-400",
                  istirahat: "red-500",
                };

                const color = statusColorMap[calender.status as "tanam" | "panen" | "istirahat"] ?? "gray-300";

                return (
                  <TableCell key={`${calender.month}-${index}`} className={`space-y-1 text-center bg-${color}`}>
                    <div className="text-xs font-medium text-muted-foreground capitalize">{calender.status}</div>
                  </TableCell>
                );
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

export { YearlyCalender };
