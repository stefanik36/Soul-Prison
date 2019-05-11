package com.stefanik36.soul_prison.util;

import com.stefanik36.soul_prison.builder.NetworkBuilder;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.network_components.network.Network;

import java.util.Random;
import java.util.function.BiFunction;

public class TestUtil {

    public static TestResult testDifferentRandoms(
            BiFunction<TrainData, Network, TestResult> testFunction,
            NetworkBuilder networkBuilder,
            TrainData trainData,
            Random random,
            int testAmount
    ) {
        TestResult result = new TestResult(0.0, 0.0);
        for (int i = 0; i < testAmount; i++) {
            Random testRandom = new Random(random.nextInt());
            networkBuilder.setRandom(testRandom);
            trainData.setRandom(testRandom);
            System.out.println("Test no. " + (i + 1));
            TestResult tr = testFunction.apply(trainData, networkBuilder.build());
            result = new TestResult(
                    result.getTestAccuracy() + tr.getTestAccuracy(),
                    result.getTestLoss() + tr.getTestLoss()
            );
        }
        return new TestResult(
                result.getTestAccuracy() / testAmount,
                result.getTestLoss() / testAmount
        );
    }
}
