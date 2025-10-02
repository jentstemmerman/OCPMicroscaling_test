# MX Float CUDA FP6 (e3m2) Example

This project provides a small CUDA C++ library that implements software conversion between 32-bit float (FP32) and a custom 6-bit floating point format with layout e3m2 (1 sign, 3 exponent bits, 2 mantissa bits).

## Format
Bits (from MSB to LSB): `s e2 e1 e0 m1 m0`
- Sign: 1 bit
- Exponent: 3 bits, bias = 3
- Mantissa: 2 bits (stored fraction bits after the implicit leading 1 for normal numbers)
- Exponent field = 0 and mantissa!=0 => subnormal
- Exponent field = 0 and mantissa=0 => zero
- Exponent field = 7 and mantissa=0 => infinity
- Exponent field = 7 and mantissa!=0 => NaN

## Build

```
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Run example
```
./example
```

## Run test
```
ctest --output-on-failure
```

## Notes
This is a purely software implementation; if future CUDA versions expose native intrinsics for 6-bit floats the conversion functions in `fp6.cu` can be updated to leverage them.
