import React, { useEffect, useState, useRef } from "react";
import { StyleSheet, Text, View, Dimensions, Platform } from "react-native";

import { Camera } from "expo-camera";

import * as tf from "@tensorflow/tfjs";
import * as posedetection from "@tensorflow-models/pose-detection";
import * as ScreenOrientation from "expo-screen-orientation";
import {
  bundleResourceIO,
  cameraWithTensors,
} from "@tensorflow/tfjs-react-native";
import Svg, { Circle, Line } from "react-native-svg";
import { ExpoWebGLRenderingContext } from "expo-gl";
import { CameraType } from "expo-camera/build/Camera.types";
import { analyzeSquat } from "./utils/sqauts";

const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === "android";
const IS_IOS = Platform.OS === "ios";

// ADDED - https://js.tensorflow.org/api_react_native/0.2.1/#Media-Camera
const texture = IS_ANDROID
  ? {
      height: 1200,
      width: 1600,
    }
  : {
      height: 1920,
      width: 1080,
    };

// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get("window").width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.3;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc). For best result, use the size that
// doesn't distort the image.
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// Whether to auto-render TensorCamera preview.
const AUTO_RENDER = true;

// Whether to load model from app bundle (true) or through network (false).
const LOAD_MODEL_FROM_BUNDLE = false;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = useState(false);
  const [model, setModel] = useState<posedetection.PoseDetector>();
  const [poses, setPoses] = useState<posedetection.Pose[]>();
  const [fps, setFps] = useState(0);
  const [orientation, setOrientation] =
    useState<ScreenOrientation.Orientation>();
  const [cameraType, setCameraType] = useState<CameraType>(CameraType.front);
  const lastAnalysisTimeRef = useRef(0);
  const [state, setState] = useState({
    result: ''
  });

  // Use `useRef` so that changing it won't trigger a re-render.
  //
  // - null: unset (initial value).
  // - 0: animation frame/loop has been canceled.
  // - >0: animation frame has been scheduled.
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Set initial orientation.
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);

      // Listens to orientation change.
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });

      // Camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
      const movenetModelConfig: posedetection.MoveNetModelConfig = {
        modelType: posedetection.SupportedModels.PoseNet,
        enableSmoothing: true,
      };
      if (LOAD_MODEL_FROM_BUNDLE) {
        const modelJson = require("./offline_model/model.json");
        const modelWeights1 = require("./offline_model/group1-shard1of2.bin");
        const modelWeights2 = require("./offline_model/group1-shard2of2.bin");
        movenetModelConfig.modelUrl = bundleResourceIO(modelJson, [
          modelWeights1,
          modelWeights2,
        ]);
      }
      const model = await posedetection.createDetector(
        posedetection.SupportedModels.PoseNet

        // movenetModelConfig
      );
      setModel(model);

      // Ready!
      setTfReady(true);
    }

    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      const now = Date.now();
      // const timeSinceLastAnalysis = now - lastAnalysisTimeRef.current;
      // if (timeSinceLastAnalysis > 500) {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value as tf.Tensor3D;

      const startTs = Date.now();
      const poses = await model!.estimatePoses(
        imageTensor,
        undefined,
        Date.now()
      );
      if (poses && poses.length) {
        try {
          const result = analyzeSquat(poses);
        setState({...state, result});
        } catch (error) {
          console.log('Error While analyzing the exercise: ', error);
        }
      }
      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      setPoses(poses);
      // if (poses && poses.length) analyzeSquat(poses);
      tf.dispose([imageTensor]);
      // lastAnalysisTimeRef.current = now;
      // }
      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false.
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      rafId.current = requestAnimationFrame(loop);
    };
    loop();
  };

  // Utility function to get adjacent keypoints
  const getAdjacentKeyPoints = (keypoints, minConfidence) => {
    const adjacentKeyPoints = [
      ["nose", "left_eye"],
      ["nose", "right_eye"],
      ["left_eye", "left_ear"],
      ["right_eye", "right_ear"],
      ["left_shoulder", "right_shoulder"],
      ["left_shoulder", "left_elbow"],
      ["left_elbow", "left_wrist"],
      ["right_shoulder", "right_elbow"],
      ["right_elbow", "right_wrist"],
      ["left_shoulder", "left_hip"],
      ["right_shoulder", "right_hip"],
      ["left_hip", "right_hip"],
      ["left_hip", "left_knee"],
      ["left_knee", "left_ankle"],
      ["right_hip", "right_knee"],
      ["right_knee", "right_ankle"],
    ];

    return adjacentKeyPoints.filter(([kp1, kp2]) => {
      const point1 = keypoints.find(
        (k) => k.name === kp1 && (k.score ?? 0) > minConfidence
      );
      const point2 = keypoints.find(
        (k) => k.name === kp2 && (k.score ?? 0) > minConfidence
      );
      return point1 && point2;
    });
  };

  const renderPose = () => {
    if (poses != null && poses.length > 0) {
      const MIN_KEYPOINT_SCORE = 0.2;

      const keypoints = poses[0].keypoints.filter(
        (k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE
      );

      const circles = keypoints.map((k) => {
        const flipX = IS_ANDROID || cameraType === CameraType.back;
        const x = flipX ? getOutputTensorWidth() - k.x : k.x;
        const y = k.y;
        const cx =
          (x / getOutputTensorWidth()) *
          (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
        const cy =
          (y / getOutputTensorHeight()) *
          (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);

        return (
          <Circle
            key={`skeletonkp_${k.name}`}
            cx={cx}
            cy={cy}
            r="4"
            strokeWidth="2"
            fill="#00AA00"
            stroke="white"
          />
        );
      });

      const SKELETON_PAIRS = getAdjacentKeyPoints(
        keypoints,
        MIN_KEYPOINT_SCORE
      );

      const lines = SKELETON_PAIRS.map(([kp1, kp2]) => {
        const p1 = keypoints.find((k) => k.name === kp1);
        const p2 = keypoints.find((k) => k.name === kp2);
        if (p1 && p2) {
          const flipX = IS_ANDROID || cameraType === CameraType.back;
          const x1 = flipX ? getOutputTensorWidth() - p1.x : p1.x;
          const y1 = p1.y;
          const x2 = flipX ? getOutputTensorWidth() - p2.x : p2.x;
          const y2 = p2.y;

          const cx1 =
            (x1 / getOutputTensorWidth()) *
            (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
          const cy1 =
            (y1 / getOutputTensorHeight()) *
            (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
          const cx2 =
            (x2 / getOutputTensorWidth()) *
            (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
          const cy2 =
            (y2 / getOutputTensorHeight()) *
            (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);

          return (
            <Line
              key={`skeleton_${kp1}_${kp2}`}
              x1={cx1}
              y1={cy1}
              x2={cx2}
              y2={cy2}
              strokeWidth="2"
              stroke="white"
            />
          );
        }
        return null;
      });

      return <Svg style={styles.svg}>{[...circles, ...lines]}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {fps}</Text>
      </View>
    );
  };

  const renderCameraTypeSwitcher = () => {
    return (
      <View
        style={styles.cameraTypeSwitcher}
        onTouchEnd={handleSwitchCameraType}
      >
        <Text>
          Switch to {cameraType === CameraType.front ? "back" : "front"} camera
        </Text>
      </View>
    );
  };

  const handleSwitchCameraType = () => {
    if (cameraType === CameraType.front) {
      setCameraType(CameraType.back);
    } else {
      setCameraType(CameraType.front);
    }
  };

  const isPortrait = () => {
    return (
      orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
      orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
    );
  };

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor to
    // get better result. Without this, the image stored in the output tensor
    // would be stretched too much.
    //
    // Same for getOutputTensorHeight below.
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };

  const getTextureRotationAngleInDegrees = () => {
    // On Android, the camera texture will rotate behind the scene as the phone
    // changes orientation, so we don't need to rotate it in TensorCamera.
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS, the camera texture won't rotate automatically. Calculate the
    // rotation angles here which will be passed to TensorCamera to rotate it
    // internally.
    switch (orientation) {
      // Not supported on iOS as of 11/2021, but add it here just in case.
      case ScreenOrientation.Orientation.PORTRAIT_DOWN:
        return 180;
      case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
        return cameraType === CameraType.front ? 270 : 90;
      case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
        return cameraType === CameraType.front ? 90 : 270;
      default:
        return 0;
    }
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      // Note that you don't need to specify `cameraTextureWidth` and
      // `cameraTextureHeight` prop in `TensorCamera` below.
      <View
        style={
          isPortrait() ? styles.containerPortrait : styles.containerLandscape
        }
      >
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          autorender={true}
          type={cameraType}
          // tensor related props
          resizeWidth={getOutputTensorWidth()}
          resizeHeight={getOutputTensorHeight()}
          resizeDepth={3}
          rotation={getTextureRotationAngleInDegrees()}
          onReady={handleCameraStream}
          // ADDED - https://js.tensorflow.org/api_react_native/0.2.1/#Media-Camera
          useCustomShadersToResize={false}
        />
        {renderPose()}
        {renderFps()}
        {renderCameraTypeSwitcher()}
        {state.result ? <Text style={{ color: "#000", position: 'absolute', bottom: -30, paddingHorizontal: 20, textTransform: 'capitalize', backgroundColor: 'red'}}>{state.result}</Text> : null}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  containerPortrait: {
    position: "relative",
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: "relative",
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: "absolute",
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  camera: {
    width: "100%",
    height: "100%",
    zIndex: 1,
  },
  svg: {
    width: "100%",
    height: "100%",
    position: "absolute",
    zIndex: 30,
  },
  fpsContainer: {
    position: "absolute",
    top: 10,
    left: 10,
    width: 80,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: "absolute",
    top: 10,
    right: 10,
    width: 180,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});
