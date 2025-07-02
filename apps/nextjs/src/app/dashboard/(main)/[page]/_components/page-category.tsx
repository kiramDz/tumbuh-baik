"use client";

import MainTable from "../../../_components/table/main-table";

interface PageFilesProps {
  category: string;
}

const PageCategory = ({ category }: PageFilesProps) => {
  return (
    <>
      <div className="container mx-auto p-0">
        <MainTable category={category} />
      </div>
    </>
  );
};

export default PageCategory;
