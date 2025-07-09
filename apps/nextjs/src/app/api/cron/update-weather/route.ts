import { NextResponse } from "next/server";

export async function GET(req: Request) {
  if (req.headers.get("Authorization") !== `Bearer ${process.env.CRON_SECRET}`) {
    return new NextResponse("Unauthorized", { status: 401 });
  }

  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/bmkg-fetch`, {
    method: "GET",
  });

  if (!res.ok) {
    return new NextResponse("Failed to trigger fetch", { status: 500 });
  }

  return NextResponse.json({ success: true });
}
