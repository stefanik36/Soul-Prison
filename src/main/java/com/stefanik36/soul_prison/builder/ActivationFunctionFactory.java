package com.stefanik36.soul_prison.builder;

import com.stefanik36.soul_prison.util.ActivationFunction;

public class ActivationFunctionFactory {

    public static ActivationFunction sigmoid() {
        return x -> 1 / (1 + Math.exp(-x));
    }

    public static ActivationFunction tanh() {
        return Math::tanh;
    }
}
