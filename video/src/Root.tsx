import { Composition, Still } from "remotion";
import { HuntAutopsyVideo, totalFrames } from "./HuntAutopsyVideo";
import { Thumbnail } from "./Thumbnail";

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
      <Still id="Thumbnail" component={Thumbnail} width={1280} height={720} />
    </>
  );
};
