@0x88f4b9e26c4cefca;

struct Camera {
  createNS @0 :UInt64;
  publishNS @1 :UInt64;
  writeNS @2 :UInt64;
  index @3 :Int8;
  jpg @4 :Data;
}

struct Action {
  createNS @0 :UInt64;
  publishNS @1 :UInt64;
  writeNS @2 :UInt64;
  isManual @3 :Bool;
  speed @4 :Float32;
  steer @5 :Float32;
}

struct Controller {
  createNS @0 :UInt64;
  publishNS @1 :UInt64;
  writeNS @2 :UInt64;
  isRecording @3 :Bool;
  isAutonomous @4 :Bool;
  speedOffset @5 :Float32;
  steerOffset @6 :Float32;
}

struct Imu {
  createNS @0 :UInt64;
  publishNS @1 :UInt64;
  writeNS @2 :UInt64;
  index @3 :Int8;
  isCalibrated @4 :Bool;
  angularVelocity @5 :List(Float32);
  magneticField @6 :List(Float32);
  linearAcceleration @7 :List(Float32);
  gravity @8 :List(Float32);
  orientationQuaternion @9 :List(Float32);
  temperature @10 :Float32;
}

struct Quality {
  createNS @0 :UInt64;
  publishNS @1 :UInt64;
  writeNS @2 :UInt64;
  quality @3 :QualityEnum;
  enum QualityEnum {
    junk @0;
    risk @1;
    good @2;
  }
}
