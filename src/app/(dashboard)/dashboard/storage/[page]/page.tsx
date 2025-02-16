import { pageIdentifier } from "@/lib/utils";
import PageFiles from "./_components/page-files";
import SubscriptionPage from "./_components/page-subscription";

interface Props {
  params: Promise<{
    page: "subscription" | "citra-satelit" | "temperatur-laut" | "daily-weather";
  }>;
}

const page = async ({ params }: Props) => {
  const page = (await params).page;
  // params.page akan mejadi valu parameter pagefiles
  const key = pageIdentifier(page);
  return (
    <>
      <h1 className="capitalize">{page}</h1>
      <br />
      {page === "subscription" ? <SubscriptionPage /> : <PageFiles category={key} />}
    </>
  );
};

export default page;
