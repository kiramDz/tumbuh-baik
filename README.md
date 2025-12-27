# 🌱 zonapetik

**zonapetik** is a **data-driven, dynamic agricultural decision support system** focused on weather analysis and planting calendar (Kalender Tanam) simulation.  
It integrates multi-source climate data, time-series forecasting, and a modular web architecture to support **evidence-based planting decisions**, especially for rice cultivation.

The system is designed to be **scalable, explainable, and modular**, suitable for research, academic evaluation, and real-world extension.

---

## 🎯 Project Goals

- Provide a **dynamic planting calendar (Kalender Tanam Dinamis)** instead of static schedules
- Utilize **time-series forecasting (Holt-Winters)** to predict rainfall and climate trends
- Integrate **multi-source climate data** (BMKG, NASA)
- Offer a **clean dashboard UI** for weather monitoring and forecast visualization
- Build a **modular backend API** that can grow as features increase

---

## 🧠 Key Features

### 🌦 Weather Dashboard

- Current weather summary
- Hourly forecast visualization
- Historical and forecasted climate charts
- Data sourced from BMKG and processed consistently

### 📅 Dynamic Planting Calendar

- Calendar generated from **forecasted rainfall patterns**
- Supports multiple planting periods (KT-1, KT-2, KT-3)
- Threshold-based evaluation (suitable / risky periods)
- Adaptable to new data without hardcoded dates

### 📈 Climate Forecasting

- Holt-Winters time-series forecasting
- Daily forecast stored and queried from database
- Forecast results used directly by planting calendar logic

### 📊 Data Management

- Dataset upload (CSV / JSON)
- Automatic collection creation in MongoDB
- Metadata tracking
- Pagination & export (CSV)

---

## 🧱 Architecture Overview

**zonapetik** follows a **clear separation of concerns**:

### Frontend

- Does **not** access the database directly
- Fetches data via REST API
- Uses React Query for:
  - Pagination
  - Caching
  - Loading & error states

### Backend API

- Built using **Hono.js** inside Next.js App Router
- Modular routing per feature (e.g. Holt-Winters)
- Clean REST endpoints under `/api/v1`

### Data Layer

- MongoDB for time-series and metadata storage
- Forecast results stored as collections and reused by UI

---

## 🧩 Backend Flow Example (Holt-Winters)

1. **UI Component**

   - Requests data using `react-query`
   - Calls `getHoltWinterDaily()`

2. **Client Fetch Layer**

   - `file.fetch.ts`
   - Handles HTTP request, params, and response normalization

3. **API Entry Point**

   - `app/api/v1/[[...route]]/route.ts`
   - Registers base API and routes using Hono

4. **Feature Route Module**
   - `holt-winter.route.ts`
   - Contains business logic:
     - Database queries
     - Pagination
     - Sorting
     - Response shaping

This approach keeps:

- UI clean
- API reusable
- Business logic centralized

---

## 🛠 Tech Stack

### Frontend

- **Next.js 15 (App Router)**
- **React**
- **TypeScript**
- **Tailwind CSS**
- **Recharts**
- **TanStack React Query**

### Backend

- **Hono.js** (modular REST API)
- **Next.js API Routes**
- **Node.js Runtime**

### Database

- **MongoDB**
- **Mongoose**

### Data & Analysis

- **Python**
- **Pandas**
- **Statsmodels (Holt-Winters)**

### Tooling

- **pnpm (monorepo)**
- **Vercel (deployment)**
- **Git & GitHub**

---

## 🔁 Why “Dynamic” Kalender Tanam?

Unlike traditional static planting calendars:

- Dates are **not hardcoded**
- Calendar updates automatically when:
  - New climate data arrives
  - Forecast results change
- Decisions are based on:
  - Forecasted rainfall patterns
  - Threshold evaluation
  - Time-series trends

This makes **zonapetik adaptive**, not descriptive.

---

## 🚀 Status

- ✅ Weather dashboard implemented
- ✅ Holt-Winters forecasting pipeline working
- ✅ Dynamic planting calendar logic ready
- 🔄 Forecast accuracy tuning (ongoing)

---

## 📌 Intended Use

- Academic research & thesis
- Agricultural decision support
- Climate-aware planning tools
- Extendable platform for future models (NDVI, ENSO, etc.)

---

If you want, I can also:

- Rewrite this README for **thesis submission**
- Make a **short abstract version**
- Add **system diagram (ASCII or Mermaid)**
- Translate it to **Bahasa Indonesia**
