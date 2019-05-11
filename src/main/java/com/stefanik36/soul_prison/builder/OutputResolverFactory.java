package com.stefanik36.soul_prison.builder;

import com.stefanik36.soul_prison.data.InOut;
import com.stefanik36.soul_prison.model.TrainSummary;
import com.stefanik36.soul_prison.util.FormatUtil;
import com.stefanik36.soul_prison.util.OutputResolver;
import io.vavr.collection.List;

import java.util.function.BiConsumer;

public class OutputResolverFactory {
    public static OutputResolver none() {
        return new OutputResolver(
                ((inOuts, trainSummary) -> {
                }),
                ((input, inOut) -> {
                }),
                (inOuts, testResult) -> {
                });
    }

    public static OutputResolver standard() {
        return new OutputResolver(
                (inOuts, trainSummary) ->
                        System.out.println("e: " + trainSummary.getEpoch()
                                + " | tLoss: " + FormatUtil.DECIMAL_FORMAT.format(trainSummary.getTrainLoss())
                                + " | vLoss: " + FormatUtil.DECIMAL_FORMAT.format(trainSummary.getValidationLoss())
                                + " | tAcc: " + FormatUtil.DECIMAL_FORMAT.format(trainSummary.getTrainAccuracy())
                                + " | vAcc: " + FormatUtil.DECIMAL_FORMAT.format(trainSummary.getValidationAccuracy())),
                (inOut, input) ->
                        System.out.println("Should be: " + inOut.getOutput().getValues().map(v -> FormatUtil.DECIMAL_FORMAT.format(v))
                                + " is: " + input.getValues().map(v -> FormatUtil.DECIMAL_FORMAT.format(v))
                        ),
                (inOuts, testResult) ->
                        System.out.println("TEST RESULT [test accuracy: " + FormatUtil.DECIMAL_FORMAT.format(testResult.getTestAccuracy())
                                + " | test loss " + FormatUtil.DECIMAL_FORMAT.format(testResult.getTestLoss()) + "]")
        );
    }

    public static OutputResolver onlyTestResults() {
        return new OutputResolver(
                (inOuts, trainSummary) -> {
                },
                (inOut, input) -> {
                },
                (inOuts, testResult) ->
                        System.out.println("TEST RESULT [test accuracy: " + FormatUtil.DECIMAL_FORMAT.format(testResult.getTestAccuracy())
                                + " | test loss " + FormatUtil.DECIMAL_FORMAT.format(testResult.getTestLoss()) + "]")
        );
    }

}
