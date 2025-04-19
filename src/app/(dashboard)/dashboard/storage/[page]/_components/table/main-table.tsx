// tempat table akan menerima data dari api, dan data column

import { useEffect, useState } from "react";
import { PaymentType } from "@/types/table-schema";
import { columns } from "./columns/temp-columns";
import { MainTableUI } from "./main-table-ui";

async function getData(): Promise<PaymentType[]> {
  const res = await fetch("https://my.api.mockaroo.com/payment_info.json?key=f0933e60");
  if (!res.ok) {
    throw new Error("Failed to fetch data");
  }

  return res.json();
}

export default function MainTable() {
  const [paymentData, setPaymentData] = useState<PaymentType[]>([]);
  useEffect(() => {
    const data = async () => {
      const result = await getData();
      setPaymentData(result);
    };
    data();
  }, []);
  return <MainTableUI columns={columns} data={paymentData} />;
}
