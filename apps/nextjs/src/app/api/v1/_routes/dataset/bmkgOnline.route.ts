import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";
import axios from "axios";
import { wrapper } from "axios-cookiejar-support";
import { CookieJar } from "tough-cookie";
import * as cheerio from "cheerio";

const bmkgOnlineRoute = new Hono();

class BMKGAuthManager {
  private cookieJar: CookieJar;
  private client: ReturnType<typeof wrapper>;
  private isAuthenticated: boolean = false;
  private lastAuthTime: number = 0;
  private authTimeout: number = 30 * 60 * 1000;

  private credentials = {
    email: process.env.BMKG_EMAIL,
    password: process.env.BMKG_PASSWORD,
  };

  constructor() {
    this.cookieJar = new CookieJar();
    this.client = wrapper(
      axios.create({
        baseURL: "https://dataonline.bmkg.go.id",
        jar: this.cookieJar as any,
        withCredentials: true,
        headers: {
          "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
          Accept:
            "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
        },
      })
    );
  }

  // Small random delay to mimic human behavior
  private async randomDelay() {
    const delay = Math.floor(Math.random() * 2000) + 500;
    await new Promise((resolve) => setTimeout(resolve, delay));
  }

  private needsAuthentication() {
    const currentTime = Date.now();
    return (
      !this.isAuthenticated ||
      currentTime - this.lastAuthTime > this.authTimeout
    );
  }
  // Get CSRF token from HTML
  private extractCsrfToken(html: string): string {
    // Try meta tag first (common in Laravel)
    const metaMatch = html.match(/<meta name="csrf-token" content="([^"]+)"/i);
    if (metaMatch && metaMatch[1]) {
      return metaMatch[1];
    }

    // Try hidden input field
    const inputMatch = html.match(/name="_token" value="([^"]+)"/i);
    if (inputMatch && inputMatch[1]) {
      return inputMatch[1];
    }

    throw new Error("CSRF token not found in page");
  }
  // Perform login to BMKG online
  async login(): Promise<boolean> {
    if (!this.needsAuthentication()) {
      console.log("Using existing BMKG session");
      return true;
    }

    try {
      console.log("Initializing new BMKG session...");
      console.log(`Using credentials for: ${this.credentials.email}`);

      // Visit homepage first to get initial cookies
      await this.client.get("/");
      console.log("Got initial cookies");
      await this.randomDelay();

      // GET login page to get CSRF token
      console.log("Requesting login page...");
      const loginPageResponse = await this.client.get("/dataonline-home");
      const html = loginPageResponse.data as string;

      // Extract CSRF token
      let csrfToken;
      try {
        csrfToken = this.extractCsrfToken(html);
        console.log(
          "Extracted CSRF token:",
          csrfToken.substring(0, 20) + "..."
        );
      } catch (err) {
        console.error("Failed to extract CSRF token:", err);
        throw err;
      }
      const $ = cheerio.load(html);
      const formFields = $("form input")
        .map((i, el) => ({
          name: $(el).attr("name"),
          type: $(el).attr("type"),
          value: $(el).attr("value"),
        }))
        .get();
      console.log("Form fields found:", formFields);

      await this.randomDelay();

      // Submit login form
      console.log("Submitting login form...");
      const loginResponse = await this.client.post(
        "/login",
        new URLSearchParams({
          _token: csrfToken,
          email: this.credentials.email || "",
          password: this.credentials.password || "",
          "g-recaptcha-response": process.env.BMKG_RECAPTCHA || "", // Try empty first
        }).toString(),
        {
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Referer: "https://dataonline.bmkg.go.id/dataonline-home",
          },
          maxRedirects: 5,
          validateStatus: (status) => {
            return status >= 200 && status < 400;
          },
        }
      );

      console.log("Login response status:", loginResponse.status);
      console.log(
        "Login response URL:",
        loginResponse.request?.res?.responseUrl
      );

      // Check if login was successful by looking at the final URL
      const finalUrl =
        loginResponse.request?.res?.responseUrl || loginResponse.config?.url;
      const isSuccess = !finalUrl?.includes("/dataonline-home");

      if (!isSuccess) {
        // Try to extract error message
        const $ = cheerio.load(loginResponse.data);
        const errorMessage = $(".alert-danger, .invalid-feedback, .error")
          .text()
          .trim();
        console.error(
          "Login failed:",
          errorMessage || "Redirected back to login page"
        );
        this.isAuthenticated = false;
        return false;
      }

      console.log("BMKG authentication successful");
      this.isAuthenticated = true;
      this.lastAuthTime = Date.now();
      return true;
    } catch (error) {
      console.error("BMKG authentication error:", error);
      if (axios.isAxiosError(error) && error.response) {
        console.error("Error response status:", error.response.status);
      }
      this.isAuthenticated = false;
      return false;
    }
  }
  async getAuthenticatedClient() {
    const isAuthenticated = await this.login();
    if (!isAuthenticated) {
      throw new Error("Failed to authenticate with BMKG");
    }
    return this.client;
  }
}
// Create the singleton instance
const bmkgAuth = new BMKGAuthManager();

// Test endpoint for login status
bmkgOnlineRoute.get("/auth-status", async (c) => {
  try {
    const isAuthenticated = await bmkgAuth.login();
    return c.json({
      success: true,
      authenticated: isAuthenticated,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return c.json(
      {
        success: false,
        message: (error as Error).message,
      },
      500
    );
  }
});

export { bmkgAuth };
export default bmkgOnlineRoute;
