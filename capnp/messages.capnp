@0x88f4b9e26c4cefca;

struct Camera {
  timeCreated @0 :Float64;
  timePublished @1 :Float64;
  timeWritten @2 :Float64;
  index @3 :Int8;
  jpg @4 :Data;
}

struct Action {
  timeCreated @0 :Float64;
  timePublished @1 :Float64;
  timeWritten @2 :Float64;
  isManual @3 :Bool;
  speed @4 :Float32;
  steer @5 :Float32;
}

struct Controller {
  timeCreated @0 :Float64;
  timePublished @1 :Float64;
  timeWritten @2 :Float64;
  isRecording @3 :Bool;
  isAutonomous @4 :Bool;
  speedOffset @5 :Float32;
  steerOffset @6 :Float32;
}

struct Imu {
  timeCreated @0 :Float64;
  timePublished @1 :Float64;
  timeWritten @2 :Float64;
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
  timeCreated @0 :Float64;
  timePublished @1 :Float64;
  timeWritten @2 :Float64;
  quality @3 :QualityEnum;
  enum QualityEnum {
    junk @0;
    risk @1;
    good @2;
  }
}
