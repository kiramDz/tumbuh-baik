import { pageIdentifier } from "@/lib/utils";
import PageFiles from "./_components/page-files";

interface Props {
  params: Promise<{
    page: "citra-satelit" | "temperatur-laut" | "daily-weather" |"bmkg-station";
  }>;
}

const page = async ({ params }: Props) => {
  const page = (await params).page;
  const key = pageIdentifier(page);
  return (
    <>
      <h1 className="capitalize">{page}</h1>
      <br />
      <PageFiles category={key} />
    </>
  );
};

export default page;
