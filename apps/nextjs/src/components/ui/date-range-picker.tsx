"use client";

import * as React from "react";
import { format } from "date-fns";
import { id } from "date-fns/locale";
import { DateRange } from "react-day-picker";
import { Icons } from "@/app/dashboard/_components/icons";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import toast from "react-hot-toast";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

type ViewMode = "date" | "month" | "year";
type PickerType = "start" | "end";

interface DateRangePickerProps {
  className?: string;
  dateRange: DateRange | undefined;
  onDateRangeChange: (range: DateRange | undefined) => void;
  disabled?: boolean;
  startPlaceholder?: string;
  endPlaceholder?: string;
}

interface SingleDatePickerProps {
  type: PickerType;
  date: Date | undefined;
  onDateSelect: (date: Date | undefined) => void;
  placeholder: string;
  disabled: boolean;
  otherDate?: Date;
}

function SingleDatePicker({
  type,
  date,
  onDateSelect,
  placeholder,
  disabled,
  otherDate,
}: SingleDatePickerProps) {
  const [open, setOpen] = React.useState(false);
  const [viewMode, setViewMode] = React.useState<ViewMode>("date");
  const [displayDate, setDisplayDate] = React.useState(date || new Date());

  // Update display date when date prop changes
  React.useEffect(() => {
    if (date) {
      setDisplayDate(date);
    }
  }, [date]);

  const formatDateString = (dateToFormat: Date | undefined) => {
    if (!dateToFormat) return "";
    return format(dateToFormat, "dd MMM yyyy", { locale: id });
  };

  const navigateMonth = (direction: "prev" | "next") => {
    if (viewMode !== "date") return;

    const newDate = new Date(displayDate);
    if (direction === "prev") {
      newDate.setMonth(newDate.getMonth() - 1);
    } else {
      newDate.setMonth(newDate.getMonth() + 1);
    }
    setDisplayDate(newDate);
  };

  const monthNames = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Mei",
    "Jun",
    "Jul",
    "Agu",
    "Sep",
    "Okt",
    "Nov",
    "Des",
  ];

  const getYearRange = (centerYear: number) => {
    const years = [];
    const startYear = Math.floor((centerYear - 2000) / 12) * 12 + 2000;
    for (let i = 0; i < 12; i++) {
      years.push(startYear + i);
    }
    return years;
  };

  const handleMonthSelect = (monthIndex: number) => {
    const newDate = new Date(displayDate);
    newDate.setMonth(monthIndex);
    setDisplayDate(newDate);
    setViewMode("date");
  };

  const handleYearSelect = (year: number) => {
    const newDate = new Date(displayDate);
    newDate.setFullYear(year);
    setDisplayDate(newDate);
    setViewMode("month");
  };

  const handleDateSelect = (selectedDate: Date | undefined) => {
    if (!selectedDate) return;

    // Validation logic based on picker type and other date
    if (otherDate) {
      if (type === "start" && selectedDate > otherDate) {
        // If start date is after end date, don't allow selection
        return;
      }
      if (type === "end" && selectedDate < otherDate) {
        // If end date is before start date, don't allow selection
        return;
      }
    }

    onDateSelect(selectedDate);
    setOpen(false);
  };

  const handleMonthClick = () => {
    setViewMode("month");
  };

  const handleYearClick = () => {
    setViewMode("year");
  };

  // Check if a date should be disabled
  const isDateDisabled = (checkDate: Date) => {
    if (!otherDate) return false;

    if (type === "start") {
      return checkDate > otherDate;
    } else {
      return checkDate < otherDate;
    }
  };

  const renderDateView = () => (
    <div>
      <div className="flex items-center justify-between p-3 border-b">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigateMonth("prev")}
          className="h-8 w-8 p-0"
        >
          <Icons.previous className="h-4 w-4" />
        </Button>

        <div className="flex items-center space-x-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleMonthClick}
            className="text-sm font-medium hover:bg-accent"
          >
            {format(displayDate, "MMMM", { locale: id })}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleYearClick}
            className="text-sm font-medium hover:bg-accent"
          >
            {displayDate.getFullYear()}
          </Button>
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigateMonth("next")}
          className="h-8 w-8 p-0"
        >
          <Icons.next className="h-4 w-4" />
        </Button>
      </div>

      <Calendar
        mode="single"
        selected={date}
        onSelect={handleDateSelect}
        month={displayDate}
        onMonthChange={setDisplayDate}
        disabled={(date) => disabled || isDateDisabled(date)}
        classNames={{
          months: "flex flex-col space-y-4",
          month: "space-y-4",
          caption: "hidden",
          table: "w-full border-collapse space-y-1",
          head_row: "flex",
          head_cell:
            "text-muted-foreground rounded-md w-8 font-normal text-[0.8rem]",
          row: "flex w-full mt-2",
          cell: "relative p-0 text-center text-sm focus-within:relative focus-within:z-20",
          day: "h-8 w-8 p-0 font-normal aria-selected:opacity-100 hover:bg-accent hover:text-accent-foreground rounded-md",
          day_selected:
            "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground focus:bg-primary focus:text-primary-foreground",
          day_today: "bg-accent text-accent-foreground",
          day_outside: "text-muted-foreground opacity-50",
          day_disabled: "text-muted-foreground opacity-50",
          day_hidden: "invisible",
        }}
      />
    </div>
  );

  const renderMonthView = () => (
    <div className="p-4">
      <div className="mb-4 text-center">
        <Button
          variant="ghost"
          onClick={handleYearClick}
          className="text-lg font-semibold hover:bg-accent"
        >
          {displayDate.getFullYear()}
        </Button>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {monthNames.map((month, index) => (
          <Button
            key={month}
            variant={displayDate.getMonth() === index ? "default" : "outline"}
            size="sm"
            onClick={() => handleMonthSelect(index)}
            className="h-10 text-sm"
          >
            {month}
          </Button>
        ))}
      </div>
    </div>
  );

  const renderYearView = () => {
    const years = getYearRange(displayDate.getFullYear());

    return (
      <div className="p-4">
        <div className="mb-4 text-center">
          <span className="text-lg font-semibold">
            {years[0]} - {years[years.length - 1]}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {years.map((year) => (
            <Button
              key={year}
              variant={
                displayDate.getFullYear() === year ? "default" : "outline"
              }
              size="sm"
              onClick={() => handleYearSelect(year)}
              className="h-10 text-sm"
            >
              {year}
            </Button>
          ))}
        </div>

        <div className="flex justify-between mt-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              const newYears = getYearRange(years[0] - 12);
              const newDate = new Date(displayDate);
              newDate.setFullYear(newYears[0]);
              setDisplayDate(newDate);
            }}
            className="text-xs"
          >
            <Icons.chevronLeft className="h-3 w-3 mr-1" />
            Prev
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              const newYears = getYearRange(years[years.length - 1] + 1);
              const newDate = new Date(displayDate);
              newDate.setFullYear(newYears[0]);
              setDisplayDate(newDate);
            }}
            className="text-xs"
          >
            Next
            <Icons.next className="h-3 w-3 ml-1" />
          </Button>
        </div>
      </div>
    );
  };

  return (
    <Popover open={open && !disabled} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start text-left font-normal",
            !date && "text-muted-foreground"
          )}
          disabled={disabled}
        >
          <Icons.calendar className="mr-2 h-4 w-4" />
          {date ? formatDateString(date) : placeholder}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-auto p-0"
        align="start"
        onOpenAutoFocus={(e) => e.preventDefault()}
      >
        {viewMode === "date" && renderDateView()}
        {viewMode === "month" && renderMonthView()}
        {viewMode === "year" && renderYearView()}
      </PopoverContent>
    </Popover>
  );
}

export function DateRangePicker({
  className,
  dateRange,
  onDateRangeChange,
  disabled = false,
  startPlaceholder = "Pilih tanggal mulai",
  endPlaceholder = "Pilih tanggal akhir",
}: DateRangePickerProps) {
  const [warningMessage, setWarningMessage] = React.useState("");

  // Handlers for start changes
  const handleStartDateChange = (startDate: Date | undefined) => {
    const newRange = {
      from: startDate,
      to: dateRange?.to,
    };

    // Check if end date is earlier than start date
    if (startDate && dateRange?.to && startDate > dateRange.to) {
      setWarningMessage(
        "Tanggal mulai tidak boleh lebih besar dari tanggal akhir"
      );
    } else {
      setWarningMessage("");
    }

    onDateRangeChange(newRange);
  };

  const handleEndDateChange = (endDate: Date | undefined) => {
    const newRange = {
      from: dateRange?.from,
      to: endDate,
    };

    // Check if end date is earlier than start date
    if (dateRange?.from && endDate && endDate < dateRange.from) {
      setWarningMessage(
        "Tanggal akhir tidak boleh lebih kecil dari tanggal mulai"
      );
    } else {
      setWarningMessage("");
    }

    onDateRangeChange(newRange);
  };

  return (
    <div className={cn("grid gap-4", className)}>
      {/* Date Range Status */}
      {dateRange?.from && dateRange?.to && (
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
          <span className="text-sm font-medium">
            Tanggal: {format(dateRange.from, "dd MMM yyyy", { locale: id })} -{" "}
            {format(dateRange.to, "dd MMM yyyy", { locale: id })}
          </span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Start Date Picker */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Tanggal Mulai *</label>
          <SingleDatePicker
            type="start"
            date={dateRange?.from}
            onDateSelect={handleStartDateChange}
            placeholder={startPlaceholder}
            disabled={disabled}
            otherDate={dateRange?.to}
          />
        </div>

        {/* End Date Picker */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Tanggal Akhir *</label>
          <SingleDatePicker
            type="end"
            date={dateRange?.to}
            onDateSelect={handleEndDateChange}
            placeholder={endPlaceholder}
            disabled={disabled}
            otherDate={dateRange?.from}
          />
        </div>
      </div>

      {/* Warning Message Display */}
      {warningMessage && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 border border-destructive/20 rounded-lg">
          <Icons.alertTriangle className="h-4 w-4 text-destructive" />
          <span className="text-sm text-destructive">{warningMessage}</span>
        </div>
      )}

      {/* Clear Button */}
      {(dateRange?.from || dateRange?.to) && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            onDateRangeChange(undefined);
            toast.success("Tanggal berhasil dihapus");
          }}
          className="w-fit self-center"
        >
          Hapus Tanggal
        </Button>
      )}
    </div>
  );
}

// Helper functions (unchanged)
export const yyyymmddToDate = (
  dateString: string | undefined
): Date | undefined => {
  if (!dateString || dateString.length !== 8) return undefined;
  const year = parseInt(dateString.substring(0, 4), 10);
  const month = parseInt(dateString.substring(4, 6), 10) - 1;
  const day = parseInt(dateString.substring(6, 8), 10);
  return new Date(year, month, day);
};

export const dateToYYYYMMDD = (date: Date | undefined): string => {
  if (!date) return "";
  const year = date.getFullYear();
  const month = (date.getMonth() + 1).toString().padStart(2, "0");
  const day = date.getDate().toString().padStart(2, "0");
  return `${year}${month}${day}`;
};
