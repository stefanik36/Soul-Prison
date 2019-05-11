package com.stefanik36.soul_prison.util;

import com.stefanik36.soul_prison.data.InOut;
import com.stefanik36.soul_prison.data.Input;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.model.TrainSummary;
import io.vavr.collection.List;

import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class OutputResolver {

    private BiConsumer<List<InOut>, TrainSummary> afterEpoch;
    private BiConsumer<InOut, Input> afterTest;
    private BiConsumer<List<InOut>, TestResult> afterAllTests;


    public OutputResolver(BiConsumer<List<InOut>, TrainSummary> afterEpoch, BiConsumer<InOut, Input> afterTest, BiConsumer<List<InOut>, TestResult> afterAllTests) {
        this.afterEpoch = afterEpoch;
        this.afterTest = afterTest;
        this.afterAllTests = afterAllTests;
    }

    public BiConsumer<List<InOut>, TrainSummary> getAfterEpoch() {
        return afterEpoch;
    }

    public void setAfterEpoch(BiConsumer<List<InOut>, TrainSummary> afterEpoch) {
        this.afterEpoch = afterEpoch;
    }

    public BiConsumer<InOut, Input> getAfterTest() {
        return afterTest;
    }

    public void setAfterTest(BiConsumer<InOut, Input> afterTest) {
        this.afterTest = afterTest;
    }

    public BiConsumer<List<InOut>, TestResult> getAfterAllTests() {
        return afterAllTests;
    }

    public void setAfterAllTests(BiConsumer<List<InOut>, TestResult> afterAllTests) {
        this.afterAllTests = afterAllTests;
    }

    public void afterEpoch(List<InOut> inOutList, TrainSummary trainSummary) {
        this.afterEpoch.accept(inOutList, trainSummary);
    }

    public void afterTest(InOut inOut, Input input) {
        this.afterTest.accept(inOut, input);
    }

    public void afterAllTests(List<InOut> inOutList, TestResult testResult) {
        this.afterAllTests.accept(inOutList, testResult);
    }
}
