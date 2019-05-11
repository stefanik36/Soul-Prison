package com.stefanik36.soul_prison.util;

import io.vavr.collection.List;

import java.util.function.BiFunction;

public interface ValidationFunction extends BiFunction<List<Double>, List<Double>, Boolean> {
}
