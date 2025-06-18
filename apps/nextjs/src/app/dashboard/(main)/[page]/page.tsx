import { pageIdentifier } from "@/lib/utils";
import PageFiles from "./_components/page-files";

interface Props {
  params: Promise<{
    page: "citra-satelit" | "temperatur-laut" | "daily-weather" |"hello";
  }>;
}

const page = async ({ params }: Props) => {
  const page = (await params).page;
  const key = pageIdentifier(page);
  return (
  <>
      <h1 className="capitalize px-5">{page}</h1>
      <br />
      <PageFiles category={key} />
    </>
  );
};

export default page;
