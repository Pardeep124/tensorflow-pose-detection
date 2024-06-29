# Overview

A modified version of [TFJS Pose Detection][posedetection] model
PoseNet using [TFJS React Native][tfjs-react-native] in an Expo project. It supports both portrait and landscape mode with front and back camera. The keypoints and lines (Skeleton) are rendered in the example.

I have used PoseNet because Movenet doesn't work properly for me. It works fine for most of the time but keypoints flicker sometimes which distorts the skeleton.
# Note

This project uses Expo SDK 49 and jsc instead of Hermes.

# Installation

To run it locally:

```
$ yarn
$ yarn start
```

Then scan the QR code to open it in the `Expo Go` app.

If the app crashes on startup, see [here][readme] for more info.

[posedetection]: https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
[tfhub]: https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4
[tfjs-react-native]: https://github.com/tensorflow/tfjs/tree/master/tfjs-react-native
[screenshots]: https://photos.app.goo.gl/U972ww4HpaKPK6jEA
[readme]: https://github.com/tensorflow/tfjs-examples/blob/master/react-native/README.md
