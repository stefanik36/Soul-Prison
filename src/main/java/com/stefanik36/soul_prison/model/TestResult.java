package com.stefanik36.soul_prison.model;

public class TestResult {
    private Double testAccuracy;
    private Double testLoss;

    public TestResult(Double testAccuracy, Double testLoss) {
        this.testAccuracy = testAccuracy;
        this.testLoss = testLoss;
    }

    public Double getTestAccuracy() {
        return testAccuracy;
    }

    public Double getTestLoss() {
        return testLoss;
    }
}
