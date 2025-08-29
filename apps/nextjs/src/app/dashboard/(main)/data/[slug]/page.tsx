import { GetDatasetBySlug } from "@/lib/fetch/files.fetch";
import { notFound } from "next/navigation";
// import { Metadata } from "next";
import DynamicMainTable from "../_components/dynamic-table";

interface Props {
  params: Promise<{ slug: string }>;
}

export default async function DatasetDetailPage({ params }: Props) {
  const { slug } = await params;
  console.log("[DEBUG] slug from route:", slug);
  const result = await GetDatasetBySlug(slug).catch(() => null);
  if (!result) return notFound();

  const { meta } = result;

  return (
    <div className="p-5">
      <h2 className="text-xl font-semibold mb-4">{meta.name}</h2>
      <DynamicMainTable
        collectionName={meta.collectionName}
        columns={meta.columns}
      />
    </div>
  );
}
