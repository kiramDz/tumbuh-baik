import { pageIdentifier } from "@/lib/utils";
import PageCategory from "./_components/page-category";

interface Props {
  params: Promise<{
    page: "citra-satelit" | "temperatur-laut" | "daily-weather" | "hello";
  }>;
}

const page = async ({ params }: Props) => {
  const page = (await params).page;
  const key = pageIdentifier(page);
  return (
    <>
      <h1 className="capitalize px-5">{page}</h1>
      <br />
      <PageCategory category={key} />
    </>
  );
};

export default page;
