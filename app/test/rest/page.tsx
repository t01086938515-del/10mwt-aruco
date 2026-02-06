"use client";

import { useRouter } from "next/navigation";
import { useAppSelector } from "@/store/hooks";
import { RestCountdown } from "@/components/test/RestCountdown";

export default function TestRestPage() {
  const router = useRouter();
  const { config, currentTrialIndex, trials } = useAppSelector(
    (state) => state.testSession
  );

  if (!config) {
    router.push("/test/setup");
    return null;
  }

  const totalTrials =
    config.mode === "both" ? config.trialsPerMode * 2 : config.trialsPerMode;

  const handleComplete = () => {
    router.push("/test/run");
  };

  return (
    <div className="min-h-screen bg-[hsl(var(--background))] p-4">
      <RestCountdown
        duration={config.restDuration}
        onComplete={handleComplete}
        trialNumber={currentTrialIndex}
        totalTrials={totalTrials}
      />
    </div>
  );
}
