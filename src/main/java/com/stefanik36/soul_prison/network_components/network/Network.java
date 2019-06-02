package com.stefanik36.soul_prison.network_components.network;

import com.stefanik36.soul_prison.data.InOut;
import com.stefanik36.soul_prison.data.Input;
import com.stefanik36.soul_prison.data.TrainData;
import com.stefanik36.soul_prison.model.TestResult;
import com.stefanik36.soul_prison.model.TrainSummary;
import com.stefanik36.soul_prison.network_components.layer.Step;
import com.stefanik36.soul_prison.network_components.layer.StepImpl;
import com.stefanik36.soul_prison.network_components.layer.StepInput;
import com.stefanik36.soul_prison.network_components.neuron.Neuron;
import com.stefanik36.soul_prison.util.OutputResolver;
import com.stefanik36.soul_prison.util.ValidationFunction;
import io.vavr.collection.List;
import io.vavr.control.Option;

public class Network {
    private static boolean showResults = true;
    private OutputResolver outputResolver;

    private StepInput inputLayer;
    private List<StepImpl> hiddenLayers;
    private StepImpl outputLayer;
    private String name;
    private long currentEpoch;
    private ValidationFunction validationFunction;

    private List<TrainSummary> trainSummaryList;
    private TestResult testResult;

    public Network(
            StepInput inputLayer,
            List<StepImpl> hiddenLayers,
            StepImpl outputLayer,
            ValidationFunction validationFunction,
            String name,
            OutputResolver outputResolver
    ) {
        this.trainSummaryList = List.empty();
        this.inputLayer = inputLayer;
        this.outputLayer = outputLayer;
        this.hiddenLayers = hiddenLayers;
        this.validationFunction = validationFunction;
        this.name = name;
        this.outputResolver = outputResolver;
    }

    public Network appendHiddenLayer(StepImpl step) {
        this.hiddenLayers = this.hiddenLayers.append(step);
        return this;
    }


    public void propagate(Input input) {
        inputLayer.setInput(input);
        List.ofAll(hiddenLayers)
                .append(outputLayer)
                .forEach(StepImpl::propagate);
    }

    public void backPropagate(Input result) {
        List<StepImpl> layers = List.ofAll(hiddenLayers)
                .append(outputLayer);

        outputLayer.computeError(result);

        for (int i : List.range(0, layers.size()).reverse()) {
            StepImpl current = layers.get(i);
            Option<StepImpl> prev = i > 0 ? Option.of(layers.get(i - 1)) : Option.none();
            current.backPropagate(prev);
        }
    }


    public void updateWeights() {
        List<StepImpl> layers = List.ofAll(hiddenLayers)
                .append(outputLayer);
        layers.forEach(StepImpl::updateWeights);
    }

    public Input getResult() {
        return new Input(outputLayer.getResults());
    }

    public List<Step> getSteps() {
        return List.of(inputLayer)
                .map(i -> (Step) i)
                .appendAll(hiddenLayers)
                .append(outputLayer);
    }

    public Double getSquareError() {
        return outputLayer.getSquareError();
    }

    public void train(TrainData data, int epochs) {
        long epochsMax = currentEpoch + epochs;
        for (; currentEpoch < epochsMax; currentEpoch++) {

            data.nextKFold();
//            if (currentEpoch % 100 == 99) {
//                this.getAllNeurons()
//                        .filter(n -> n instanceof NeuronNode)
//                        .map(n -> (NeuronNode) n)
//                        .forEach(nn -> nn.setLearningRate(nn.getLearningRate() / 10));
//            }
            List<InOut> trainData = data.getTrain();
            List<InOut> validationData = data.getValidation();
            trainEpoch(trainData, validationData);
        }
    }


    public void test(TrainData data) {
        doTest(data.getTest());
    }


    public void trainAutoEncoder(TrainData data, int epochs) {
        long epochsMax = currentEpoch + epochs;
        for (; currentEpoch < epochsMax; currentEpoch++) {

            data.nextKFold();
            List<InOut> trainData = data.getAutoEncoderTrain();
            List<InOut> validationData = data.getAutoEncoderValidation();

            trainEpoch(trainData, validationData);
        }
    }

    public void testAutoEncoder(TrainData data) {
        doTest(data.getAutoEncoderTest());
    }


    private void trainEpoch(List<InOut> trainData, List<InOut> validationData) {
        double tSumSquareError = 0.0;
        int tValid = 0;

        for (InOut td : trainData) {
            this.propagate(td.getInput());
            tSumSquareError += this.getSquareError();
            tValid += isValid(td.getOutput(), this.getResult()) ? 1 : 0;
            this.backPropagate(td.getOutput());
            this.updateWeights();
        }

        double vSumSquareError = 0.0;
        int vValid = 0;

        for (InOut vd : validationData) {
            this.propagate(vd.getInput());
            vSumSquareError += this.getSquareError();
            vValid += isValid(vd.getOutput(), this.getResult()) ? 1 : 0;
        }

        TrainSummary ts = new TrainSummary(
                currentEpoch,
                (double) tValid / trainData.size(),
                tSumSquareError / trainData.size(),
                (double) vValid / validationData.size(),
                vSumSquareError / validationData.size());

        this.trainSummaryList = this.trainSummaryList.append(ts);

        this.outputResolver.afterEpoch(validationData, ts);
    }

    private boolean isValid(Input realResult, Input netResult) {
        return validationFunction.apply(realResult.getValues(), netResult.getValues());
    }


    public List<TrainSummary> getTrainSummaryList() {
        return trainSummaryList;
    }


    private void doTest(List<InOut> testData) {
        double tSumSquareError = 0.0;
        int tValid = 0;

        for (InOut td : testData) {
            this.propagate(td.getInput());
            Input result = this.getResult();
            tSumSquareError += this.getSquareError();
            boolean isValid = isValid(td.getOutput(), this.getResult());
            tValid += isValid ? 1 : 0;

            outputResolver.afterTest(td, result);
        }

        double testAccuracy = (double) tValid / (double) testData.size();
        double testLoss = tSumSquareError / (double) testData.size();
        this.testResult = new TestResult(testAccuracy, testLoss);

        outputResolver.afterAllTests(testData, testResult);
    }

    public TestResult getTestResult() {
        return testResult;
    }

    public List<Neuron> getAllNeurons() {
        return getSteps().flatMap(Step::getNeuronsWithBias);
    }
}




























