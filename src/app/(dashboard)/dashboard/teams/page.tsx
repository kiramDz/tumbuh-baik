// pake design : https://dribbble.com/shots/22585345-Member-Area-CRM-AI-Integration
// function : add people
// name | email | role | joined | status
import PageContainer from "@/components/page-container";

export default function TeamLayout() {
  return (
    <>
      <PageContainer>
        <div className="flex flex-1 flex-col space-y-2">
          <div className="flex items-center justify-between space-y-2">
            <h2 className="text-2xl font-bold tracking-tight">Team Member 👥</h2>
          </div>
        </div>
      </PageContainer>
    </>
  );
}
