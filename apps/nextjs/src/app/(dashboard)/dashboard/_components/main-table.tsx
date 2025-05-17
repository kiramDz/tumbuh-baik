"use client";

import { columns } from "./columns";
import { MainTableUI } from "./main-table-ui";

const dummyData = [
  {
    source: "BMKG Station",
    date: "2025-04-28",
    record: 120,
    status: "backlog",
  },
  {
    source: "Buoys",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "Citra Satelite",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "Daily weather",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "BMKG Station",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "BMKG Station",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "BMKG Station",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "BMKG Station",
    date: "2025-04-28",
    record: 120,
    status: "in progress",
  },
  {
    source: "Citra Satelit",
    date: "2025-04-27",
    record: 95,
    status: "backlog",
  },
  {
    source: "Temperatur Laut",
    date: "2025-04-26",
    record: 0,
    status: "canceled",
  },
  {
    source: "Daily Weather",
    date: "2025-04-25",
    record: 110,
    status: "backlog",
  },
];

export default function MainTable() {
  return <MainTableUI columns={columns} data={dummyData} />;
}
