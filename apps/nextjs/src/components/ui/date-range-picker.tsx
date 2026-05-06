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

  // Max date is today (no future dates allowed)
  const today = new Date();
  today.setHours(0, 0, 0, 0);

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

  const navigateYearRange = (direction: "prev" | "next") => {
    const newDate = new Date(displayDate);
    if (direction === "prev") {
      newDate.setFullYear(newDate.getFullYear() - 12);
    } else {
      newDate.setFullYear(newDate.getFullYear() + 12);
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

    // Don't allow future dates
    if (selectedDate > today) {
      toast.error("Tidak bisa memilih tanggal yang belum terjadi");
      return;
    }

    // Validation logic based on picker type and other date
    if (otherDate) {
      if (type === "start" && selectedDate > otherDate) {
        toast.error("Tanggal mulai tidak boleh lebih besar dari tanggal akhir");
        return;
      }
      if (type === "end" && selectedDate < otherDate) {
        toast.error("Tanggal akhir tidak boleh lebih kecil dari tanggal mulai");
        return;
      }
    }

    onDateSelect(selectedDate);
    setOpen(false);
  };

  const handleTodayClick = () => {
    handleDateSelect(today);
  };

  const handleMonthClick = () => {
    setViewMode("month");
  };

  const handleYearClick = () => {
    setViewMode("year");
  };

  // Check if a date should be disabled
  const isDateDisabled = (checkDate: Date) => {
    // Disable future dates
    if (checkDate > today) return true;

    if (!otherDate) return false;

    if (type === "start") {
      return checkDate > otherDate;
    } else {
      return checkDate < otherDate;
    }
  };

  // Check if month should be disabled (all days in month are disabled)
  const isMonthDisabled = (monthIndex: number) => {
    const testDate = new Date(displayDate.getFullYear(), monthIndex, 1);

    // Disable if month is in the future
    if (
      testDate.getFullYear() > today.getFullYear() ||
      (testDate.getFullYear() === today.getFullYear() &&
        monthIndex > today.getMonth())
    ) {
      return true;
    }

    return false;
  };

  // Check if year should be disabled
  const isYearDisabled = (year: number) => {
    return year > today.getFullYear();
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
          disabled={
            displayDate.getFullYear() === today.getFullYear() &&
            displayDate.getMonth() === today.getMonth()
          }
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
          day_today: "bg-accent text-accent-foreground font-semibold",
          day_outside: "text-muted-foreground opacity-50",
          day_disabled: "text-muted-foreground opacity-50 cursor-not-allowed",
          day_hidden: "invisible",
        }}
      />

      {/* Today Button */}
      <div className="p-3 border-t">
        <Button
          variant="outline"
          size="sm"
          onClick={handleTodayClick}
          className="w-full"
        >
          Hari Ini
        </Button>
      </div>
    </div>
  );

  const renderMonthView = () => (
    <div className="p-4">
      <div className="mb-4 flex items-center justify-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            const newDate = new Date(displayDate);
            newDate.setFullYear(newDate.getFullYear() - 1);
            setDisplayDate(newDate);
          }}
          disabled={displayDate.getFullYear() <= 2000}
          className="h-8 w-8 p-0"
        >
          <Icons.chevronLeft className="h-4 w-4" />
        </Button>

        <Button
          variant="ghost"
          onClick={handleYearClick}
          className="text-lg font-semibold hover:bg-accent min-w-[100px]"
        >
          {displayDate.getFullYear()}
        </Button>

        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            const newDate = new Date(displayDate);
            newDate.setFullYear(newDate.getFullYear() + 1);
            setDisplayDate(newDate);
          }}
          disabled={displayDate.getFullYear() >= today.getFullYear()}
          className="h-8 w-8 p-0"
        >
          <Icons.next className="h-4 w-4" />
        </Button>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {monthNames.map((month, index) => {
          const isDisabled = isMonthDisabled(index);
          const isCurrent = displayDate.getMonth() === index;

          return (
            <Button
              key={month}
              variant={isCurrent ? "default" : "outline"}
              size="sm"
              onClick={() => handleMonthSelect(index)}
              disabled={isDisabled}
              className={cn(
                "h-10 text-sm",
                isDisabled && "opacity-50 cursor-not-allowed",
              )}
            >
              {month}
            </Button>
          );
        })}
      </div>
    </div>
  );

  const renderYearView = () => {
    const years = getYearRange(displayDate.getFullYear());

    return (
      <div className="p-4">
        <div className="mb-4 flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigateYearRange("prev")}
            disabled={years[0] <= 2000}
            className="text-xs"
          >
            <Icons.chevronLeft className="h-3 w-3 mr-1" />
            {years[0] - 12} - {years[0] - 1}
          </Button>

          <span className="text-lg font-semibold">
            {years[0]} - {years[years.length - 1]}
          </span>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigateYearRange("next")}
            disabled={years[years.length - 1] >= today.getFullYear()}
            className="text-xs"
          >
            {years[years.length - 1] + 1} - {years[years.length - 1] + 12}
            <Icons.next className="h-3 w-3 ml-1" />
          </Button>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {years.map((year) => {
            const isDisabled = isYearDisabled(year);
            const isCurrent = displayDate.getFullYear() === year;

            return (
              <Button
                key={year}
                variant={isCurrent ? "default" : "outline"}
                size="sm"
                onClick={() => handleYearSelect(year)}
                disabled={isDisabled}
                className={cn(
                  "h-10 text-sm",
                  isDisabled && "opacity-50 cursor-not-allowed",
                )}
              >
                {year}
              </Button>
            );
          })}
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
            !date && "text-muted-foreground",
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
  // Handlers for start changes
  const handleStartDateChange = (startDate: Date | undefined) => {
    const newRange = {
      from: startDate,
      to: dateRange?.to,
    };
    onDateRangeChange(newRange);
  };

  const handleEndDateChange = (endDate: Date | undefined) => {
    const newRange = {
      from: dateRange?.from,
      to: endDate,
    };
    onDateRangeChange(newRange);
  };

  return (
    <div className={cn("grid gap-4", className)}>
      {/* Date Range Status */}
      {dateRange?.from && dateRange?.to && (
        <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border">
          <div className="flex items-center gap-2">
            <Icons.calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">
              {format(dateRange.from, "dd MMM yyyy", { locale: id })} -{" "}
              {format(dateRange.to, "dd MMM yyyy", { locale: id })}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              onDateRangeChange(undefined);
              toast.success("Tanggal berhasil dihapus");
            }}
            className="h-8 px-2 text-xs"
          >
            <Icons.closeX className="h-3 w-3" />
          </Button>
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

      {/* Info Message */}
      {!dateRange?.from && !dateRange?.to && (
        <div className="flex items-start gap-2 p-3 bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900 rounded-lg">
          <Icons.info className="h-4 w-4 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
          <span className="text-sm text-blue-800 dark:text-blue-300">
            Pilih tanggal mulai dan akhir untuk menentukan periode
          </span>
        </div>
      )}
    </div>
  );
}

// Helper functions (unchanged)
export const yyyymmddToDate = (
  dateString: string | undefined,
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
