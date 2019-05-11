package com.stefanik36.soul_prison.model;

public class TrainSummary {
    private long epoch;

    private Double trainAccuracy;
    private Double trainLoss;

    private Double validationAccuracy;
    private Double validationLoss;

    public TrainSummary(
            long epoch,
            Double trainAccuracy,
            Double trainLoss,
            Double validationAccuracy,
            Double validationLoss
    ) {
        this.epoch = epoch;
        this.trainAccuracy = trainAccuracy;
        this.trainLoss = trainLoss;
        this.validationAccuracy = validationAccuracy;
        this.validationLoss = validationLoss;
    }

    public long getEpoch() {
        return epoch;
    }

    public Double getTrainAccuracy() {
        return trainAccuracy;
    }

    public Double getTrainLoss() {
        return trainLoss;
    }

    public Double getValidationAccuracy() {
        return validationAccuracy;
    }

    public Double getValidationLoss() {
        return validationLoss;
    }
}
