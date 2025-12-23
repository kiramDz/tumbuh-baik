"use client";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import React, { useState } from "react";
import {
  AddDatasetMeta,
  fetchNasaPowerData,
  saveNasaPowerData,
  AddXlsxDatasetMeta,
} from "@/lib/fetch/files.fetch";
import { parseFile } from "@/lib/parse-upload";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import toast from "react-hot-toast";
import { Icons } from "@/app/dashboard/_components/icons";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { fi } from "date-fns/locale";
import { DateRange } from "react-day-picker";
import { FileUploader } from "@/components/ui/file-uploader";
import {
  DateRangePicker,
  dateToYYYYMMDD,
  yyyymmddToDate,
} from "@/components/ui/date-range-picker";

type LocationEntry = {
  name: string;
  lat: number;
  lon: number;
};

type AcehLocations = {
  [kabupaten: string]: LocationEntry[];
};

const ACEH_LOCATIONS: AcehLocations = {
  "Kabupaten Aceh Besar": [
    { name: "Kec. Indrapuri", lat: 5.4218918, lon: 95.4463322 },
    { name: "Kec. Montasik", lat: 5.4803, lon: 95.4594 },
    { name: "Kec. Darussalam", lat: 5.5945451, lon: 95.4201377 },
  ],
  "Kabupaten Aceh Jaya": [
    { name: "Kec. Jaya/Lamno", lat: 5.1526721, lon: 95.1509617 },
    { name: "Kec. Setia Bakti", lat: 4.8333007, lon: 95.4867971 },
    { name: "Kec. Pasie Raya", lat: 4.6136123, lon: 95.8614504 },
    { name: "Kec. Teunom", lat: 4.4999535, lon: 95.7700208 },
  ],
  "Kabupaten Pidie": [
    { name: "Kec. Pidie", lat: 5.3759998, lon: 95.9148038 },
    { name: "Kec. Indrajaya", lat: 5.3114261, lon: 95.8978653 },
  ],
  "Kabupaten Aceh Utara": [
    { name: "Kec. Lhoksukon", lat: 5.051701, lon: 97.318123 },
  ],
  "Kabupaten Bireuen": [
    { name: "Kec. Juli", lat: 5.0735373, lon: 96.5879472 },
    { name: "Kec. Kota Juang", lat: 5.190849, lon: 96.6728368 },
  ],
};

const PARAMETER_OPTIONS = [
  // Temperature parameters
  { value: "T2M", label: "Temperature at 2 Meters (°C)" },
  { value: "T2M_MAX", label: "Maximum Temperature at 2 Meters (°C)" },
  { value: "T2M_MIN", label: "Minimum Temperature at 2 Meters (°C)" },

  // Humidity parameter
  { value: "RH2M", label: "Relative Humidity at 2 Meters (%)" },

  // Precipitation parameter
  { value: "PRECTOTCORR", label: "Precipitation (mm/day)" },

  // Solar radiation parameters
  {
    value: "ALLSKY_SFC_SW_DWN",
    label: "All Sky Surface Shortwave Downward Irradiance (W/m^2)",
  },

  // Wind parameters
  { value: "WS10M", label: "Wind Speed at 10 Meters (m/s)" },
  { value: "WS10M_MAX", label: "Maximum Wind Speed at 10 Meters (m/s)" },
  { value: "WD10M", label: "Wind Direction at 10 Meters (degrees)" },
];

const nasaPowerSchema = z.object({
  name: z.string().min(3, "Name must be at least 3 characters"),
  description: z.string().optional(),
  // Allow empty string for initial state, but validate format when not empty
  start: z.string().refine((val) => val === "" || /^\d{8}$/.test(val), {
    message: "Format must be YYYYMMDD",
  }),
  end: z.string().refine((val) => val === "" || /^\d{8}$/.test(val), {
    message: "Format must be YYYYMMDD",
  }),
  latitude: z.number().min(-90).max(90),
  longitude: z.number().min(-180).max(180),
  parameters: z.array(z.string()).min(1, "Select at least one parameter"),
});

type NasaPowerFormValues = z.infer<typeof nasaPowerSchema>;

export default function AddDatasetDialog() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState("upload");
  const [open, setOpen] = useState(false);
  const [dateWarningMessage, setDateWarningMessage] = useState("");
  const [locationWarningMessage, setLocationWarningMessage] = useState("");
  const [selectedKabupaten, setSelectedKabupaten] = useState<string>("");
  const [selectedKecamatan, setSelectedKecamatan] = useState<string>("");

  // Upload Form state
  const [uploadForm, setUploadForm] = useState({
    name: "",
    source: "",
    collectionName: "",
    description: "",
    status: "raw",
  });

  // Multi-file support state
  const [isMultiFile, setIsMultiFile] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [files, setFiles] = useState<File[]>([]);

  // NASA POWER form state
  const [previewData, setPreviewData] = useState<any>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState(false);

  // NASA POWER form
  const nasaPowerForm = useForm<NasaPowerFormValues>({
    resolver: zodResolver(nasaPowerSchema),
    defaultValues: {
      name: "",
      description: "",
      start: "",
      end: "",
      latitude: 0,
      longitude: 0,
      parameters: PARAMETER_OPTIONS.map((param) => param.value),
    },
  });

  // Reset form include multi-file upload
  const resetForm = () => {
    setOpen(false);
    setUploadForm({
      name: "",
      source: "",
      collectionName: "",
      description: "",
      status: "raw",
    });
    setFile(null);
    setFiles([]);
    setIsMultiFile(false);
    nasaPowerForm.reset();
    setPreviewData(null);
    setDateWarningMessage("");
    setSelectedKabupaten("Pilih Kabupaten");
    setSelectedKecamatan("");
    setDateRange(undefined);
  };

  // State for data range
  const [dateRange, setDateRange] = useState<DateRange | undefined>(undefined);

  React.useEffect(() => {
    // Set default Kecamatan based on initial coordinates
    const initialLat = nasaPowerForm.getValues("latitude");
    const initialLng = nasaPowerForm.getValues("longitude");

    for (const [kabupaten, locations] of Object.entries(ACEH_LOCATIONS)) {
      const matchingLocation = locations.find(
        (loc: LocationEntry) =>
          Math.abs(loc.lat - initialLat) < 0.001 &&
          Math.abs(loc.lon - initialLng) < 0.001
      );
      if (matchingLocation) {
        setSelectedKabupaten(kabupaten);
        setSelectedKecamatan(matchingLocation.name);
        break;
      }
    }
  }, [nasaPowerForm]);

  // Handle date range change
  const handleDateRangeChange = (range: DateRange | undefined) => {
    setDateRange(range);

    // Check date validation and set warning
    if (range?.from && range?.to) {
      if (range.from > range.to) {
        setDateWarningMessage(
          "Tanggal mulai tidak boleh lebih besar dari tanggal akhir"
        );
      } else {
        setDateWarningMessage("");
      }
    } else {
      setDateWarningMessage("");
    }

    // Update form values based on selected range
    if (range?.from) {
      nasaPowerForm.setValue("start", dateToYYYYMMDD(range.from));
    }
    if (range?.to) {
      nasaPowerForm.setValue("end", dateToYYYYMMDD(range.to));
    }
  };

  //Upload mutation to handle CSV/JSON and also XLSX
  const { mutate: uploadMutate, isPending: uploadPending } = useMutation({
    mutationKey: ["add-dataset"],
    mutationFn: async (uploadData: any) => {
      // Check if it's XLSX (single or multi)
      if (uploadData.fileType === "xlsx") {
        return AddXlsxDatasetMeta(uploadData); // XLSX upload
      } else {
        return AddDatasetMeta(uploadData); // CSV/JSON uploads
      }
    },
    onSuccess: () => {
      toast.success(`Dataset ${uploadForm.name} berhasil ditambahkan!`);
      queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
      resetForm();
    },
    onError: (
      error: Error & { response?: { data?: { message?: string } } }
    ) => {
      toast.error(error.response?.data?.message || "Gagal menyimpan dataset");
    },
  });

  // Handle Upload form submit
  const handleUploadSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Basic validation
    if (!uploadForm.name || !uploadForm.source) {
      return toast.error("Mohon lengkapi nama dan sumber data");
    }

    // Validate file selection based on mode
    if (isMultiFile) {
      if (files.length === 0) {
        return toast.error("Pilih minimal 1 file untuk upload multi-file");
      }
      if (files.length > 12) {
        return toast.error("Maksimum 12 file yang dapat dipilih");
      }
      // Check if all files are XLSX for multi-file mode
      const nonXlsxFiles = files.filter(
        (f) => !f.name.toLowerCase().endsWith(".xlsx")
      );
      if (nonXlsxFiles.length > 0) {
        return toast.error("Mode multi-file hanya mendukung file XLSX");
      }
    } else {
      if (!file) {
        return toast.error("Pilih file untuk diupload");
      }
    }

    // ✅ Determine file type based on extension
    const fileName = isMultiFile ? files[0].name : file!.name;
    let fileType: "csv" | "json" | "xlsx";

    if (fileName.toLowerCase().endsWith(".xlsx")) {
      fileType = "xlsx";
    } else if (fileName.toLowerCase().endsWith(".json")) {
      fileType = "json";
    } else if (fileName.toLowerCase().endsWith(".csv")) {
      fileType = "csv";
    } else {
      return toast.error("Hanya file CSV, JSON, atau XLSX yang didukung");
    }

    // ✅ Handle XLSX differently from CSV/JSON
    if (fileType === "xlsx") {
      toast.promise(
        new Promise((resolve, reject) => {
          uploadMutate(
            {
              name: uploadForm.name,
              source: uploadForm.source,
              description: uploadForm.description,
              status: uploadForm.status,
              // ✅ Pass the appropriate file(s) based on mode
              ...(isMultiFile
                ? { files: files, isMultiFile: true }
                : { file: file, isMultiFile: false }),
              fileType: "xlsx",
            },
            {
              onSuccess: () => {
                queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
                resetForm();
                resolve("success");
              },
              onError: (error) => {
                reject(error);
              },
            }
          );
        }),
        {
          loading: isMultiFile
            ? `Mengkonversi dan menggabungkan ${files.length} file XLSX...`
            : "Mengkonversi dan menyimpan file XLSX...",
          success: `Dataset ${uploadForm.name} berhasil disimpan${
            isMultiFile ? ` dari ${files.length} file` : ""
          }!`,
          error: (err) =>
            `${err?.response?.data?.message || "Gagal menyimpan dataset"}`,
        }
      );
      return;
    }

    const buffer = await file!.arrayBuffer();
    const parsed = await parseFile({
      fileBuffer: Buffer.from(buffer),
      fileType,
    });

    toast.promise(
      new Promise((resolve, reject) => {
        uploadMutate(
          {
            name: uploadForm.name,
            source: uploadForm.source,
            fileType,
            description: uploadForm.description,
            status: uploadForm.status,
            records: parsed,
          },
          {
            onSuccess: () => {
              queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
              resetForm();
              resolve("success");
            },
            onError: (error) => {
              reject(error);
            },
          }
        );
      }),
      {
        loading: "Menyimpan dataset...",
        success: `Dataset ${uploadForm.name} berhasil disimpan!`,
        error: (err) =>
          `${err?.response?.data?.message || "Gagal menyimpan dataset"}`,
      }
    );
  };
  // Handle Preview NASA POWER data
  const handleNasaPowerPreview = async () => {
    const validationErrors = [];

    // Check for Kecamatan selection
    if (!selectedKecamatan) {
      validationErrors.push("Pilih Kecamatan terlebih dahulu!");
    }

    if (!dateRange?.from) {
      validationErrors.push("Pilih tanggal mulai terlebih dahulu!");
    }

    if (!dateRange?.to) {
      validationErrors.push("Pilih tanggal akhir terlebih dahulu!");
    }

    // If there are validation errors, show them all and stop
    if (validationErrors.length > 0) {
      // Show each error as a separate toast
      validationErrors.forEach((error) => {
        toast.error(error);
      });
      return;
    }

    try {
      setPreviewLoading(true);
      const values = nasaPowerForm.getValues();
      const response = await fetchNasaPowerData({
        start: values.start,
        end: values.end,
        latitude: values.latitude,
        longitude: values.longitude,
        parameters: values.parameters,
      });
      setPreviewData(response.data);

      toast.success(`Preview data berhasil dimuat!`);
    } catch (error: any) {
      toast.error(error.message || "Gagal memuat preview data");
      console.error(error);
    } finally {
      setPreviewLoading(false);
    }
  };

  // Also update the handleNasaPowerSubmit function similarly
  const handleNasaPowerSubmit = async (values: NasaPowerFormValues) => {
    const validationErrors = [];
    // Check for name length
    if (!values.name || values.name.length < 3) {
      validationErrors.push("Nama dataset minimal 3 karakter!");
    }

    // Check for location
    if (!selectedKecamatan) {
      validationErrors.push("Pilih Kecamatan terlebih dahulu!");
    }

    // Add validation for date range
    if (!dateRange?.from) {
      validationErrors.push("Pilih tanggal mulai terlebih dahulu!");
    }

    if (!dateRange?.to) {
      validationErrors.push("Pilih tanggal akhir terlebih dahulu!");
    }
    // If there are validation errors, show them all and stop
    if (validationErrors.length > 0) {
      validationErrors.forEach((error) => {
        toast.error(error);
      });
      return;
    }

    toast.promise(
      new Promise(async (resolve, reject) => {
        try {
          setSaveLoading(true);
          await saveNasaPowerData({
            name: values.name,
            description: values.description,
            status: "raw",
            source: "Data NASA (https://power.larc.nasa.gov/)",
            nasaParams: {
              start: values.start,
              end: values.end,
              latitude: values.latitude,
              longitude: values.longitude,
              parameters: values.parameters,
            },
          });
          queryClient.invalidateQueries({ queryKey: ["dataset-meta"] });
          resetForm();
          setSaveLoading(false);
          resolve("success");
        } catch (error) {
          setSaveLoading(false);
          reject(error);
        }
      }),
      {
        loading: "Menyimpan dataset NASA POWER...",
        success: `Dataset ${values.name} berhasil disimpan!`,
        error: (err) =>
          `${err?.response?.data?.message || "Gagal menyimpan dataset!"}`,
      }
    );
  };
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        setOpen(value);
        if (!value) resetForm();
      }}
    >
      <DialogTrigger asChild>
        <Button
          variant="outline"
          className="border-green-600 text-green-700 hover:bg-green-50 hover:border-green-700 font-semibold px-5 py-2 rounded-lg shadow-sm transition-colors"
        >
          Dataset Baru
        </Button>
      </DialogTrigger>
      <DialogContent className="w-[95vw] max-w-[600px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Tambah Dataset</DialogTitle>
          <DialogDescription>
            Unggah file CSV/JSON/XLSX atau ambil data dari NASA POWER API.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="mt-4">
          <TabsList className="grid w-full grid-cols-1 sm:grid-cols-2 p-1 bg-muted/30">
            <TabsTrigger
              value="upload"
              className={`
      transition-all duration-200 rounded-md
      ${
        activeTab === "upload"
          ? "bg-black text-white font-medium shadow-md" // Changed to black with white text
          : "hover:bg-muted/80"
      }
    `}
            >
              <div className="flex items-center gap-2">
                <Icons.fileUpload className="h-4 w-4" />{" "}
                {/* Always show icon */}
                <span>Upload File</span>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="nasa-power"
              className={`
      transition-all duration-200 rounded-md
      ${
        activeTab === "nasa-power"
          ? "bg-black text-white font-medium shadow-md" // Changed to black with white text
          : "hover:bg-muted/80"
      }
    `}
            >
              <div className="flex items-center gap-2">
                <Icons.cloud className="h-4 w-4" /> {/* Always show icon */}
                <span>NASA POWER API</span>
              </div>
            </TabsTrigger>
          </TabsList>

          {/* Upload File Tab */}
          <TabsContent value="upload">
            <form onSubmit={handleUploadSubmit} className="grid gap-4 py-4">
              {/* ✅ Multi-file toggle checkbox */}
              <div className="grid gap-2">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="multiFile"
                    checked={isMultiFile}
                    onCheckedChange={(checked) => {
                      setIsMultiFile(!!checked);
                      // Reset file selections when toggling
                      setFile(null);
                      setFiles([]);
                    }}
                  />
                  <Label htmlFor="multiFile" className="text-sm font-medium">
                    Upload Multi-File XLSX (untuk menggabungkan beberapa file)
                  </Label>
                </div>
                {isMultiFile && (
                  <div className="text-xs text-muted-foreground bg-blue-50 border border-blue-200 rounded p-2">
                    <p className="font-medium text-blue-700 mb-1">
                      Catatan Multi-File Upload:
                    </p>
                    <ul className="space-y-1 text-blue-600">
                      <li className="flex items-start gap-1">
                        <span>•</span>
                        <span>
                          Mode multi-file mendukung file XLSX - digabungkan
                          menjadi satu dataset
                        </span>
                      </li>
                      <li className="flex items-start gap-1">
                        <span>•</span>
                        <span>
                          Pastikan sumber dataset sama, dan dalam rentang 12
                          bulan
                        </span>
                      </li>
                    </ul>
                  </div>
                )}
              </div>

              <div className="grid gap-2">
                <Label htmlFor="name">Nama Dataset *</Label>
                <Input
                  id="name"
                  value={uploadForm.name}
                  onChange={(e) =>
                    setUploadForm({ ...uploadForm, name: e.target.value })
                  }
                  required
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="source">Sumber</Label>
                <select
                  id="source"
                  value={uploadForm.source}
                  onChange={(e) =>
                    setUploadForm({ ...uploadForm, source: e.target.value })
                  }
                  className="border rounded px-3 py-2"
                  required
                >
                  <option value="">Pilih sumber data...</option>
                  <option value="Data BMKG (https://dataonline.bmkg.go.id/)">
                    Data BMKG (https://dataonline.bmkg.go.id/)
                  </option>
                  <option value="Data NASA (https://power.larc.nasa.gov/)">
                    Data NASA (https://power.larc.nasa.gov/)
                  </option>
                  <option value="Data BUOYS (https://www.pmel.noaa.gov)">
                    Data BUOYS (https://www.pmel.noaa.gov)
                  </option>
                </select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="status">Status</Label>
                <select
                  id="status"
                  value={uploadForm.status}
                  onChange={(e) =>
                    setUploadForm({ ...uploadForm, status: e.target.value })
                  }
                  className="border rounded px-3 py-2"
                >
                  <option value="raw">Raw</option>
                  <option value="cleaned">Cleaned</option>
                  <option value="validated">Validated</option>
                </select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="description">Deskripsi</Label>
                <Input
                  id="description"
                  value={uploadForm.description}
                  onChange={(e) =>
                    setUploadForm({
                      ...uploadForm,
                      description: e.target.value,
                    })
                  }
                />
              </div>

              {/* ✅ Updated File Upload Section */}
              <div className="grid gap-2">
                <Label>
                  {isMultiFile ? "Upload Multiple Files (XLSX)" : "Upload File"}
                </Label>
                <FileUploader
                  accept={isMultiFile ? ".xlsx" : ".csv,.json,.xlsx"}
                  maxSize={50}
                  maxTotalSize={100}
                  maxFiles={12}
                  multiple={isMultiFile}
                  onFileSelect={isMultiFile ? undefined : setFile}
                  onFilesSelect={isMultiFile ? setFiles : undefined}
                  selectedFile={isMultiFile ? undefined : file}
                  selectedFiles={isMultiFile ? files : undefined}
                  loading={uploadPending}
                />

                {/* ✅ File type indicators */}
                {isMultiFile && files.length > 0 && (
                  <div className="text-xs text-muted-foreground mt-1">
                    <span className="text-blue-600 font-medium">
                      📊 {files.length} file Excel akan digabungkan dan
                      dikonversi ke format CSV
                    </span>
                  </div>
                )}

                {!isMultiFile && file && (
                  <div className="text-xs text-muted-foreground mt-1">
                    {file.name.toLowerCase().endsWith(".xlsx") ? (
                      <span className="text-blue-600 font-medium">
                        📊 File Excel akan dikonversi ke format CSV
                      </span>
                    ) : file.name.toLowerCase().endsWith(".json") ? (
                      <span className="text-green-600 font-medium">
                        📄 File JSON siap diproses
                      </span>
                    ) : (
                      <span className="text-orange-600 font-medium">
                        📈 File CSV siap diproses
                      </span>
                    )}
                  </div>
                )}

                {/* ✅ Multi-file processing indicator */}
                {uploadPending && isMultiFile && (
                  <div className="mt-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="flex items-center gap-2 text-blue-700 text-sm">
                      <Icons.spinner className="h-4 w-4 animate-spin" />
                      <span>
                        Sedang mengkonversi dan menggabungkan {files.length}{" "}
                        file XLSX...
                      </span>
                    </div>
                    <div className="text-xs text-blue-600 mt-1">
                      Proses penggabungan mungkin membutuhkan waktu untuk banyak
                      file
                    </div>
                  </div>
                )}
              </div>

              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="outline">Batal</Button>
                </DialogClose>
                <Button type="submit" disabled={uploadPending}>
                  {uploadPending ? (
                    <>
                      <Icons.spinner className="h-4 w-4 animate-spin mr-2" />
                      {isMultiFile
                        ? "Menggabungkan..."
                        : file?.name.toLowerCase().endsWith(".xlsx")
                        ? "Mengkonversi..."
                        : "Menyimpan..."}
                    </>
                  ) : (
                    "Simpan"
                  )}
                </Button>
              </DialogFooter>
            </form>
          </TabsContent>

          {/* NASA POWER API Tab */}
          <TabsContent value="nasa-power">
            <Form {...nasaPowerForm}>
              <form
                onSubmit={nasaPowerForm.handleSubmit(handleNasaPowerSubmit)}
                className="space-y-4 py-4"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <FormField
                    control={nasaPowerForm.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Nama Dataset*</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="NASA Aceh Besar Kec. Indrapuri 2005-2025"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={nasaPowerForm.control}
                    name="description"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Deskripsi (Opsional)</FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder="Deskripsi singkat tentang dataset ini"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
                <div className="space-y-4">
                  <FormField
                    control={nasaPowerForm.control}
                    name="latitude"
                    render={() => (
                      <FormItem>
                        <FormLabel>Lokasi*</FormLabel>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <FormLabel className="text-sm font-normal text-muted-foreground mb-2">
                              Kabupaten
                            </FormLabel>
                            <select
                              className="w-full border rounded px-3 py-2 text-sm"
                              value={selectedKabupaten}
                              onChange={(e) => {
                                const kabupaten = e.target.value;
                                setSelectedKabupaten(kabupaten);
                                setSelectedKecamatan("");

                                // Reset coordinates when kabupaten changes
                                nasaPowerForm.setValue("latitude", 0);
                                nasaPowerForm.setValue("longitude", 0);
                              }}
                            >
                              <option value="">Pilih Kabupaten</option>
                              {Object.keys(ACEH_LOCATIONS).map((kabupaten) => (
                                <option key={kabupaten} value={kabupaten}>
                                  {kabupaten}
                                </option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <FormLabel className="text-sm font-normal text-muted-foreground mb-2">
                              Kecamatan
                            </FormLabel>
                            <select
                              className="w-full border rounded px-3 py-2 text-sm"
                              value={selectedKecamatan}
                              onChange={(e) => {
                                const kecamatan = e.target.value;
                                setSelectedKecamatan(kecamatan);

                                if (!kecamatan) {
                                  // Reset coordinates if no kecamatan selected
                                  setLocationWarningMessage(
                                    "Silahkan pilih Kecamatan"
                                  );
                                  nasaPowerForm.setValue("latitude", 0);
                                  nasaPowerForm.setValue("longitude", 0);
                                  return;
                                }

                                setLocationWarningMessage("");

                                // Find selected location data
                                const location = ACEH_LOCATIONS[
                                  selectedKabupaten
                                ]?.find((loc) => loc.name === kecamatan);

                                // Update form with coordinates
                                if (location) {
                                  nasaPowerForm.setValue(
                                    "latitude",
                                    location.lat
                                  );
                                  nasaPowerForm.setValue(
                                    "longitude",
                                    location.lon
                                  );
                                }
                              }}
                              disabled={!selectedKabupaten} // Disable if no kabupaten is selected
                            >
                              <option value="">Pilih Kecamatan</option>
                              {selectedKabupaten &&
                              ACEH_LOCATIONS[selectedKabupaten]
                                ? ACEH_LOCATIONS[selectedKabupaten].map(
                                    (location) => (
                                      <option
                                        key={location.name}
                                        value={location.name}
                                      >
                                        {location.name}
                                      </option>
                                    )
                                  )
                                : null}
                            </select>
                          </div>
                        </div>
                        {/* Show the selected coordinates (read-only) */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
                          <div>
                            <FormLabel className="text-xs text-muted-foreground">
                              Latitude
                            </FormLabel>
                            <div className="text-sm font-medium">
                              {selectedKecamatan
                                ? nasaPowerForm.watch("latitude")
                                : "-"}
                            </div>
                          </div>
                          <div>
                            <FormLabel className="text-xs text-muted-foreground">
                              Longitude
                            </FormLabel>
                            <div className="text-sm font-medium">
                              {selectedKecamatan
                                ? nasaPowerForm.watch("longitude")
                                : "-"}
                            </div>
                          </div>
                        </div>

                        {/* Hidden inputs to store actual latitude/longitude values */}
                        <input
                          type="hidden"
                          {...nasaPowerForm.register("latitude", {
                            setValueAs: (v) => parseFloat(v),
                          })}
                        />
                        <input
                          type="hidden"
                          {...nasaPowerForm.register("longitude", {
                            setValueAs: (v) => parseFloat(v),
                          })}
                        />

                        <FormMessage>
                          {nasaPowerForm.formState.errors.latitude?.message ||
                            nasaPowerForm.formState.errors.longitude?.message}
                        </FormMessage>
                      </FormItem>
                    )}
                  />
                </div>

                <div className="space-y-2 mb-4">
                  <FormField
                    control={nasaPowerForm.control}
                    name="start"
                    render={() => (
                      <FormItem className="space-y-2">
                        <FormControl>
                          <DateRangePicker
                            dateRange={dateRange}
                            onDateRangeChange={handleDateRangeChange}
                            disabled={previewLoading || saveLoading}
                          />
                        </FormControl>

                        {/* Add this to display date-specific warnings */}
                        {dateWarningMessage && (
                          <p className="text-amber-500 text-sm mt-1">
                            {dateWarningMessage}
                          </p>
                        )}

                        {/* Hidden inputs for form validation */}
                        <input
                          type="hidden"
                          {...nasaPowerForm.register("start")}
                        />
                        <input
                          type="hidden"
                          {...nasaPowerForm.register("end")}
                        />

                        <FormMessage>
                          {nasaPowerForm.formState.errors.start?.message ||
                            nasaPowerForm.formState.errors.end?.message}
                        </FormMessage>
                      </FormItem>
                    )}
                  />
                </div>

                {/* <FormField
                  control={nasaPowerForm.control}
                  name="parameters"
                  render={() => (
                    <FormItem>
                      <div className="mb-2">
                        <FormLabel>Parameter (pilih manual)</FormLabel>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {PARAMETER_OPTIONS.map((option) => (
                          <div
                            key={option.value}
                            className="flex items-center space-x-2"
                          >
                            <Checkbox
                              checked={nasaPowerForm
                                .watch("parameters")
                                .includes(option.value)}
                              onCheckedChange={(checked) => {
                                const current =
                                  nasaPowerForm.getValues("parameters");
                                if (checked) {
                                  nasaPowerForm.setValue("parameters", [
                                    ...current,
                                    option.value,
                                  ]);
                                } else {
                                  nasaPowerForm.setValue(
                                    "parameters",
                                    current.filter(
                                      (value) => value !== option.value
                                    )
                                  );
                                }
                              }}
                              id={`param-${option.value}`}
                            />
                            <label
                              htmlFor={`param-${option.value}`}
                              className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                            >
                              {option.label}
                            </label>
                          </div>
                        ))}
                      </div>
                      <FormMessage />
                    </FormItem>
                  )}
                /> */}
                {previewData && (
                  <Card>
                    <CardContent className="p-4">
                      <h3 className="font-semibold mb-2">Data Preview</h3>
                      <div className="max-h-40 overflow-y-auto text-xs">
                        <pre>{JSON.stringify(previewData, null, 2)}</pre>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <DialogFooter className="flex flex-col sm:flex-row justify-between gap-2 pt-4">
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={handleNasaPowerPreview}
                    disabled={previewLoading || saveLoading}
                    className="flex items-center gap-2 transition-all hover:bg-secondary/80"
                  >
                    {previewLoading ? (
                      <>
                        <Icons.spinner className="h-4 w-4 animate-spin" />
                        <span>Loading...</span>
                      </>
                    ) : (
                      <>
                        <Icons.preview className="h-4 w-4" />
                        <span>Preview</span>
                      </>
                    )}
                  </Button>
                  <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
                    <DialogClose asChild>
                      <Button variant="outline">Batal</Button>
                    </DialogClose>
                    <Button
                      type="submit"
                      disabled={previewLoading || saveLoading}
                      className="flex items-center gap-2 transition-colors"
                    >
                      {saveLoading ? (
                        <>
                          <Icons.spinner className="h-4 w-4 animate-spin" />
                          <span>Menyimpan...</span>
                        </>
                      ) : (
                        <>
                          <Icons.save className="h-4 w-4" />
                          <span>Simpan</span>
                        </>
                      )}
                    </Button>
                  </div>
                </DialogFooter>
              </form>
            </Form>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
