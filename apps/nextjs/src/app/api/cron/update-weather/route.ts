import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  // Vercel cron jobs mengirim Authorization header secara otomatis
  const authHeader = req.headers.get("Authorization");
  const expectedAuth = `Bearer ${process.env.CRON_SECRET}`;

  if (authHeader !== expectedAuth) {
    return new NextResponse("Unauthorized", { status: 401 });
  }

  try {
    const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/bmkg-fetch`, {
      method: "GET",
    });

    if (!res.ok) {
      console.error("Failed to trigger fetch:", res.status, res.statusText);
      return new NextResponse("Failed to trigger fetch", { status: 500 });
    }

    console.log("Cron job executed successfully");
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Cron job error:", error);
    return new NextResponse("Internal Server Error", { status: 500 });
  }
}
