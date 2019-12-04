@0x81f5c310643f2eca;

struct Input {
  timestamp @0 :UInt64;
  speed @1 :Float32;
  steer @2 :Float32;
  offset_speed @3 :Float32;
  offset_steer @4 :Float32;
  record @5 :Bool;
  auto @6 :Bool;
}
