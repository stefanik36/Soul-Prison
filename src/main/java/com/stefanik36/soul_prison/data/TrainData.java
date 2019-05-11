package com.stefanik36.soul_prison.data;

import com.stefanik36.soul_prison.util.ListUtil;
import io.vavr.collection.List;

import java.util.Random;

public class TrainData {
    private static final double TEST_PERCENT = 0.3;
    private static final double VALIDATION_PERCENT = 0.2;
    private int test;
    private int validation;
    private Random random;

    private int iterator;

    private List<InOut> inOutList;
    private List<InOut> testData;
    private List<InOut> trainAndValidationData;
    private List<InOut> validationData;
    private List<InOut> trainData;

    public TrainData(List<InOut> inOutList, Random random) {
        this.inOutList = inOutList;
        this.random = random;
        this.test = ((Double) (inOutList.size() * TEST_PERCENT)).intValue();
        this.validation = ((Double) (inOutList.size() * VALIDATION_PERCENT)).intValue();
        updateData();
    }

    public TrainData(List<DataResultTuple> iData, Random random, List<Double> categoryValues) {
        this(iData.map(dt -> new InOut(new Input(dt.getData()).getNormalized(), new Input(dt.getClassificationResult(categoryValues)))), random);
    }

    private void updateData() {
        this.inOutList = ListUtil.shuffle(inOutList, random);
        this.testData = this.inOutList.subSequence(0, test);
        this.trainAndValidationData = this.inOutList.subSequence(test);
    }

    public TrainData setTest(double testFraction) {
        this.test = ((Double) (inOutList.size() * testFraction)).intValue();
        updateData();
        return this;
    }

    public TrainData setValidation(double validationFraction) {
        this.validation = ((Double) (inOutList.size() * validationFraction)).intValue();
        return this;
    }

    public void setRandom(Random random) {
        this.random = random;
        updateData();
    }

    public void kFold() {
        if (iterator + validation >= this.trainAndValidationData.size()) {
            iterator = 0;
            ListUtil.shuffle(this.trainAndValidationData, random);
        }
        this.validationData = this.trainAndValidationData.subSequence(iterator, iterator + validation);
        this.trainData = this.trainAndValidationData.removeAll(this.validationData);
        iterator = iterator + validation;
    }

    public List<InOut> getTest() {
        return testData;
    }

    public List<InOut> getTrain() {
        return this.trainData;
    }

    public List<InOut> getValidation() {
        return this.validationData;
    }

    public List<InOut> getAutoEncoderTrain() {
        return this.trainData.map(td -> new InOut(td.getInput(), td.getInput()));
    }

    public List<InOut> getAutoEncoderValidation() {
        return this.validationData.map(td -> new InOut(td.getInput(), td.getInput()));
    }

    public List<InOut> getAutoEncoderTest() {
        return this.testData.map(td -> new InOut(td.getInput(), td.getInput()));
    }
}
