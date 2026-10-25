import { Composition } from "remotion";
import { HuntAutopsyVideo, totalFrames } from "./HuntAutopsyVideo";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="HuntAutopsy"
        component={HuntAutopsyVideo}
        durationInFrames={totalFrames(30)}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
