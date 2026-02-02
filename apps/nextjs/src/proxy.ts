import { NextResponse, type NextRequest } from "next/server";
import { getSessionCookie } from "better-auth/cookies";

// Helper function to get user role from database
async function getUserRole(req: NextRequest) {
  try {
    const response = await fetch(`${req.nextUrl.origin}/api/auth/get-session`, {
      headers: {
        cookie: req.headers.get("cookie") || "",
      },
      // Tambahkan cache: 'no-store' untuk memastikan tidak menggunakan cache
      cache: "no-store",
    });

    if (!response.ok) return null;

    const session = await response.json();

    // Pastikan session benar-benar valid dan memiliki user
    if (!session || !session.user || !session.session) {
      return null;
    }

    return session.user.role || null;
  } catch (error) {
    console.error("Error getting user role:", error);
    return null;
  }
}

export const config = {
  matcher: ["/dashboard/:path*", "/admin/:path*", "/sign-in", "/sign-up"],
};

export default async function authProxy(req: NextRequest) {
  const { pathname, search } = req.nextUrl;
  const sessionCookie = getSessionCookie(req);

  // Public routes - always allow
  if (pathname === "/") {
    return NextResponse.next();
  }

  // Handle sign-up page
  if (pathname === "/sign-up") {
    if (sessionCookie) {
      // Verify session is actually valid, not just cookie exists
      const userRole = await getUserRole(req);
      if (userRole === "admin") {
        return NextResponse.redirect(new URL("/dashboard", req.url));
      } else if (userRole) {
        // Valid user but not admin
        return NextResponse.redirect(new URL("/", req.url));
      }
      // If userRole is null, session is invalid, continue to sign-up
    }
    return NextResponse.next();
  }

  // Handle sign-in page
  if (pathname === "/sign-in") {
    if (sessionCookie) {
      // Verify session is actually valid, not just cookie exists
      const userRole = await getUserRole(req);
      if (userRole === "admin") {
        return NextResponse.redirect(new URL("/dashboard", req.url));
      } else if (userRole) {
        // Valid user but not admin
        return NextResponse.redirect(new URL("/", req.url));
      }
      // If userRole is null, session is invalid, continue to sign-in
    }
    return NextResponse.next();
  }

  // Protected routes - require authentication
  if (pathname.startsWith("/dashboard") || pathname.startsWith("/admin")) {
    if (!sessionCookie) {
      // Not logged in, redirect to sign-in
      return NextResponse.redirect(new URL(`/sign-in?next=${pathname}${search}`, req.url));
    }

    // Get user role for authorization
    const userRole = await getUserRole(req);

    // If session is invalid (userRole is null), redirect to login
    if (!userRole) {
      return NextResponse.redirect(new URL(`/sign-in?next=${pathname}${search}`, req.url));
    }

    // Dashboard access - only admins
    if (pathname.startsWith("/dashboard")) {
      if (userRole !== "admin") {
        // Regular user trying to access dashboard, redirect to home
        return NextResponse.redirect(new URL("/", req.url));
      }
    }

    // Admin routes - only admins (if you have /admin routes)
    if (pathname.startsWith("/admin")) {
      if (userRole !== "admin") {
        return NextResponse.redirect(new URL("/", req.url));
      }
    }

    return NextResponse.next();
  }

  return NextResponse.next();
}
