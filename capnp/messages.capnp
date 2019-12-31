@0x88f4b9e26c4cefca;

struct Camera {
  timeCreated @0 :UInt64;
  timePublished @1 :UInt64;
  timeWritten @2 :UInt64;
  yaw @3 :Float32;
  pitch @4 :Float32;
  roll @5 :Float32;
  x @6 :Float32;
  y @7 :Float32;
  z @8 :Float32;
  height @9 :UInt16;
  width @10 :UInt16;
  depth @11 :UInt8;
  hfov @12 :Float32;
  vfov @13 :Float32;
  fps @14 :UInt8;
  jpg @15 :Data;
}

struct Label {
  timeCreated @0 :UInt64;
  timePublished @1 :UInt64;
  timeWritten @2 :UInt64;
  quality @3 :QualityEnum;
  enum QualityEnum {
    unknown @0;
    trash @1;
    risky @2;
    good @3;
  }
}

struct Control {
  timeCreated @0 :UInt64;
  timePublished @1 :UInt64;
  timeWritten @2 :UInt64;
  speed @3 :Float32;
  steer @4 :Float32;
  manual @5 :Bool;
}

struct State {
  timeCreated @0 :UInt64;
  timePublished @1 :UInt64;
  timeWritten @2 :UInt64;
  record @3 :Bool;
  auto @4 :Bool;
  speedOffset @5 :Float32;
  steerOffset @6 :Float32;
}

struct Imu {
  timeCreated @0 :UInt64;
  timePublished @1 :UInt64;
  timeWritten @2 :UInt64;
  calibrationGyroscope @3 :UInt8;
  calibrationAccelerometer @4 :UInt8;
  calibrationMagnetometer @5 :UInt8;
  accelerometer @6 :List(Float32);
  calibration @7 :Data;
  gravity @8 :List(Float32);
  gyroscope @9 :List(Float32);
  magnetometer @10 :List(Float32);
  quaternion @11 :List(Float32);
}
