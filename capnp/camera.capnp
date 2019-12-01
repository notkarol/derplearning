@0x88f4b9e26c4cefca;

struct Camera {
  timestamp @0 :UInt64;
  yaw @1 :Float32;
  pitch @2 :Float32;
  roll @3 :Float32;
  x @4 :Float32;
  y @5 :Float32;
  z @6 :Float32;
  height @7 :UInt16;
  width @8 :UInt16;
  depth @9 :UInt8;
  hfov @10 :Float32;
  vfov @11 :Float32;
  fps @12 :UInt8;
  jpg @13 :Data;
}
