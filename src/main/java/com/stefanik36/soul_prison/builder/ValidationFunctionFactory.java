package com.stefanik36.soul_prison.builder;

import com.stefanik36.soul_prison.util.ValidationFunction;

public class ValidationFunctionFactory {

    public static ValidationFunction classification() {
        return (real, net) -> net.indexOf(net.max().get()) == real.indexOf(real.max().get());
    }

    public static ValidationFunction binary() {
        return (real, net) -> real
                .zipWith(
                        net,
                        (r, n) -> ((r >= 0.5 && n >= 0.5) || (r < 0.5 && n < 0.5))
                )
                .reject(v -> v)
                .isEmpty();
    }


    public static ValidationFunction equalWithAccuracy(double accuracy) {
        return (real, net) -> real
                .zipWith(
                        net,
                        (r, n) -> Math.abs(r - n) < accuracy
                )
                .reject(v -> v)
                .isEmpty();
    }
}
