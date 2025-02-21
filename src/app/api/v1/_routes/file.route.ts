import { getServerSession } from "@/action/auth.action";
import db from "@/lib/database/db";
import { File } from "@/lib/database/schema/file.model";
import { Subscription } from "@/lib/database/schema/subscription.model";
import { pinata } from "@/lib/pinata/config";
import { parseError } from "@/lib/utils";
import { Hono } from "hono";

const fileRoute = new Hono();

fileRoute.get("/", async (c) => {
  try {
    await db();
    const session = await getServerSession();
    const search = c.req.query("search");

    if (!session) {
      return c.json(
        {
          message: "Unauthorized",
          description: "You need to be logged in to upload files",
        },
        { status: 401 }
      );
    }

    if (!search || search.trim() === "") {
      return c.json(
        {
          message: "Bad Request",
          description: "Search term cannot be empty.",
        },
        { status: 400 }
      );
    }

    const {
      user: { id },
    } = session;

    const files = await File.find({
      "userInfo.id": id,
      name: {
        $regex: search,
        $options: "i",
      },
    }).lean();

    return c.json(
      {
        message: "Successful",
        description: "",
        data: files,
      },
      { status: 200 }
    );
  } catch (error) {
    console.log("Error in searching file: ", error);

    const err = parseError(error);

    return c.json({ message: "Error", description: err }, { status: 500 });
  }
});

fileRoute.get("/recent", async (c) => {
  try {
    await db(); // Pastikan koneksi ke database
    console.log("=== Endpoint /recent terpanggil ===");

    // Cari 10 file terbaru, hanya ambil field yang diperlukan
    const recentFiles = await File.find({})
      .select("name category size userInfo") // userInfo berisi nama owner
      .sort({ createdAt: -1 })
      .limit(10)
      .lean();

    return c.json({
      message: "Success",
      description: "",
      data: {
        files: recentFiles,
        total: recentFiles.length,
      },
    });
  } catch (error) {
    console.error("Error fetching recent files:", error);
    const err = parseError(error);
    return c.json({ message: "Error", description: err }, { status: 500 });
  }
});


// endpoint untuk membuat halaman khushs untuk image | doc | pdf dll
// fileRoute.get("/:page", async (c) => {
//   try {
//     await db();
//     const category = c.req.param("page");
//     const page = Number(c.req.query("page"));
//     const session = await getServerSession();
//     const FILE_SIZE = 9;

//     if (!session) {
//       return c.json(
//         {
//           message: "Unauthorized",
//           description: "You need to be logged in to upload files",
//         },
//         {
//           status: 401,
//         }
//       );
//     }

//     const {
//       user: { id: userId, email: userEmail },
//     } = session;

//     if (category === "shared") {
//       const documentCount = await File.aggregate([{ $unwind: "$sharedWith" }, { $match: { "sharedWith.email": userEmail } }, { $count: "totalDocuments" }]);

//       const totalFiles = documentCount.length > 0 ? documentCount[0].totalDocuments : 0;

//       const files = await File.aggregate([
//         { $unwind: "$sharedWith" },
//         { $match: { "sharedWith.email": userEmail } },
//         {
//           $group: {
//             _id: "$_id", // Group back the files by their original ID
//             pinataId: { $first: "$pinataId" },
//             name: { $first: "$name" },
//             cid: { $first: "$cid" },
//             size: { $first: "$size" },
//             mimeType: { $first: "$mimeType" },
//             userInfo: { $first: "$userInfo" },
//             groupId: { $first: "$groupId" },
//             sharedWith: { $push: "$sharedWith" }, // Reconstruct the sharedWith array
//             category: { $first: "$category" },
//             createdAt: { $first: "$createdAt" },
//             updatedAt: { $first: "$updatedAt" },
//           },
//         },
//       ]);

//       return c.json(
//         {
//           message: "Success",
//           description: "",
//           data: {
//             files: files,
//             total: totalFiles,
//             currentPage: page,
//             totalPages: Math.ceil(totalFiles / FILE_SIZE),
//           },
//         },
//         { status: 200 }
//       );
//     }

//     const totalFiles = await File.countDocuments({
//       "userInfo.id": userId,
//       category,
//     });

//     const files = await File.find({ "userInfo.id": userId, category })
//       .skip((page - 1) * FILE_SIZE)
//       .limit(FILE_SIZE)
//       .sort({ createdAt: -1 })
//       .lean();

//     return c.json(
//       {
//         message: "Success",
//         description: "",
//         data: {
//           files: files,
//           total: totalFiles,
//           currentPage: page,
//           totalPages: Math.ceil(totalFiles / FILE_SIZE),
//         },
//       },
//       { status: 200 }
//     );
//   } catch (error) {
//     console.log("Error in fetching files: ", error);
//     const err = parseError(error);

//     return c.json(
//       {
//         message: "Error",
//         description: err,
//         data: null,
//       },
//       { status: 500 }
//     );
//   }
// });

// endpoint untuk membuat halaman base on category : citra-satelite | buoys dll
fileRoute.get("/:category", async (c) => {
  try {
    console.log("=== Endpoint /CATEGORY terpanggil ===");
    await db();
    const category = c.req.param("category");
    const page = Number(c.req.query("page")) || 1;
    const FILE_SIZE = 9;

    const totalFiles = await File.countDocuments({
      category,
    });

    const files = await File.find({ category })
      .skip((page - 1) * FILE_SIZE)
      .limit(FILE_SIZE)
      .sort({ createdAt: -1 })
      .lean();

    return c.json(
      {
        message: "Success",
        description: "",
        data: {
          files,
          total: totalFiles,
          currentPage: page,
          totalPages: Math.ceil(totalFiles / FILE_SIZE),
        },
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("Error in fetching category files: ", error);
    return c.json(
      {
        message: "Error",
        description: "Failed to fetch files",
        data: null,
      },
      { status: 500 }
    );
  }
});

fileRoute.post("/upload", async (c) => {
  try {
    // pastikan koneksi dengan database
    await db();

    const data = await c.req.formData();
    // const file: File | null = data.get("file") as unknown as File;
    // Ambil file dari request (data.get("file")).
    const file = data.get("file") as File | null;
    if (!file) {
      return c.json({ message: "No file provided" }, { status: 400 });
    }
    console.log("Received file:", file);
    console.log("File type:", file?.type);

    const session = await getServerSession();

    if (!session) {
      return c.json(
        {
          file: null,
          message: "Unauthorized",
          description: "You need to be logged in to upload files",
        },
        {
          status: 401,
        }
      );
    }

    const userId = session.user.id;
    const name = session.user.name;

    const subs = await Subscription.findOne({ subscriber: userId });

    if (!subs) {
      return c.json(
        {
          message: "⚠️ Warning",
          category: null,
          description: "Subscription not found. Please log out and log in again to refresh your session.",
          file: null,
        },
        { status: 404 }
      );
    }

    if (subs.subscriptionType !== "free" && subs.status !== "activated") {
      return c.json(
        {
          message: "⚠️ Warning",
          category: null,
          description: "Your subscription has expired. Please re-subscribe to continue.",
          file: null,
        },
        { status: 400 }
      );
    }

    if (subs.selectedStorage <= subs.usedStorage) {
      return c.json(
        {
          message: "⚠️ Warning",
          category: null,
          description: "Storage limit exceeded. Please subscribe and select additional storage.",
          file: null,
        },
        { status: 400 }
      );
    }
    // Kategorisasi File, periksa fortmat file dengan `getCategoryFromMimeType` for detail
    const category = data.get("category") as string;
    console.log("Received category:", category);
    if (!["bmkg-station", "citra-satelit", "temperatur-laut", "daily-weather"].includes(category)) {
      return c.json({ message: "Invalid category" }, { status: 400 });
    }
    console.log("Received category:", category);
    const categoryMap: Record<string, string> = {
      "bmkg-station": "BMKG",
      "citra-satelit": "Citra Satelit",
      "temperatur-laut": "Temperatur Laut",
      "daily-weather": "Daily Weather",
    };

    const mappedCategory = categoryMap[category] || category;
    // unggah File ke Pinata
    // **Upload file ke Pinata hanya setelah semua validasi berhasil**
    const uploadData = await pinata.upload.file(file).addMetadata({
      keyvalues: { userId, name },
    });

    // Menyimpan Informasi File ke Database
    const uploadedFile = await File.create({
      pinataId: uploadData.id,
      name: uploadData.name,
      mimeType: uploadData.mime_type,
      cid: uploadData.cid,
      size: uploadData.size,
      userInfo: { id: userId, name },
      category: mappedCategory,
    });

    await Subscription.updateOne(
      { subscriber: userId },
      {
        $inc: {
          usedStorage: uploadData.size,
        },
      }
    );

    return c.json(
      {
        message: "Upload Successful",
        category,
        description: `File: ${uploadData.name}`,
        file: uploadedFile,
      },
      { status: 201 }
    );
  } catch (error) {
    console.log("Error in file uploading: ", error);

    const err = parseError(error);

    return c.json({ message: "Error", description: err, file: null }, { status: 500 });
  }
});

//==================== TES RECENT 2 :
fileRoute.get("/recent2", async (c) => {
  try {
    console.log("=== Endpoint /recent2 terpanggil ===");
    await db();

    const recentFiles = await File.find().sort({ createdAt: -1 }).limit(10);
    console.log("=== Data yang dikirim ke frontend ===");
    console.log(recentFiles); // Debugging

    return c.json({ message: "Success", data: { files: recentFiles, total: recentFiles.length } }); // Pastikan selalu mengembalikan array
  } catch (error) {
    console.error("Error fetching recent files:", error);
    return c.json({ files: [], message: "Error", description: "Failed to fetch recent files" }, { status: 500 });
  }
});

// =================== RECENT ENDPOINT GAGAL

export default fileRoute;
