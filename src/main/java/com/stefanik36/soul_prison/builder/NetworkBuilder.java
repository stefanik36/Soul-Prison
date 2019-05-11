package com.stefanik36.soul_prison.builder;

import com.stefanik36.soul_prison.network_components.layer.Step;
import com.stefanik36.soul_prison.network_components.layer.StepImpl;
import com.stefanik36.soul_prison.network_components.layer.StepInput;
import com.stefanik36.soul_prison.network_components.network.Network;
import com.stefanik36.soul_prison.util.ActivationFunction;
import com.stefanik36.soul_prison.util.OutputResolver;
import com.stefanik36.soul_prison.util.ValidationFunction;
import io.vavr.collection.List;
import io.vavr.control.Option;

import java.util.Random;

public class NetworkBuilder {
    private static final Double MEAN = 0.0;
    private static final Double RANGE = 1.0;
    private static final Double MOMENTUM = 0.4;
    private static final Double LEARNING_RATE = 0.3;
    private static final String NAME_PREFIX = "";
    private static final Double BIAS = 1.0;

    private static int build = 0;

    enum BuilderType {
        FULLY_CONNECTED, FROM_STEPS;
    }

    private Double bias;
    private Double mean;
    private Double range;
    private Double momentum;
    private Double learningRate;

    private Random random;
    private String name;
    private ActivationFunction activationFunction;
    private ValidationFunction validationFunction;

    private BuilderType builderType;
    private int inputSize;
    private int outputSize;
    private List<Integer> hiddenLayersSizes;
    private List<Step> steps;
    private OutputResolver outputResolver;

    private NetworkBuilder(
            BuilderType builderType,
            ValidationFunction validationFunction,
            int inputSize,
            int outputSize,
            List<Integer> hiddenLayersSizes,
            List<Step> steps
    ) {
        this.bias = BIAS;
        this.mean = MEAN;
        this.range = RANGE;
        this.momentum = MOMENTUM;
        this.learningRate = LEARNING_RATE;

        this.name = "net_" + build;
        this.random = new Random();
        this.activationFunction = ActivationFunctionFactory.sigmoid();
        this.outputResolver = OutputResolverFactory.none();

        this.builderType = builderType;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayersSizes = hiddenLayersSizes;
        this.validationFunction = validationFunction;

        this.steps = steps;
    }


    private NetworkBuilder(BuilderType builderType, ValidationFunction validationFunction, int inputSize, int outputSize, List<Integer> hiddenLayersSizes) {
        this(builderType, validationFunction, inputSize, outputSize, hiddenLayersSizes, null);
    }

    private NetworkBuilder(BuilderType builderType, ValidationFunction validationFunction, List<Step> steps) {
        this(builderType, validationFunction, 0, 0, null, steps);
    }


    /*
     * Initialize methods
     */

    public static NetworkBuilder initFullyConnected(ValidationFunction validationFunction, int inputSize, int outputSize, int... hiddenLayersSizes) {
        return new NetworkBuilder(BuilderType.FULLY_CONNECTED, validationFunction, inputSize, outputSize, List.ofAll(hiddenLayersSizes));
    }

    public static NetworkBuilder initFromSteps(ValidationFunction validationFunction, Step... steps) {
        return initFromSteps(validationFunction, List.of(steps));
    }

    public static NetworkBuilder initFromSteps(ValidationFunction validationFunction, List<Step> steps) {
        return new NetworkBuilder(BuilderType.FROM_STEPS, validationFunction, steps);
    }


    /*
     * Build methods
     */

    public Network build() {
        Network result;
        switch (builderType) {
            case FULLY_CONNECTED:
                result = buildFullyConnected();
                break;
            case FROM_STEPS:
                result = buildFromSteps();
                break;
            default:
                throw new UnsupportedOperationException();
        }
        build++;
        return result;
    }

    private Network buildFromSteps() {
        return new Network(
                (StepInput) steps.get(0),
                steps.subSequence(1, steps.size() - 1).map(s -> (StepImpl) s),
                (StepImpl) steps.last(),
                validationFunction,
                this.name,
                this.outputResolver
        );
    }

    private Network buildFullyConnected() {
        StepInput stepInput = StepBuilder
                .initInput(inputSize)
                .setBias(bias)
                .setNamePrefix(this.name + " ")
                .buildInput();

        List<StepImpl> stepHiddenList = List.empty();
        Step in = stepInput;
        for (int i = 0; i < hiddenLayersSizes.size(); i++) {
            StepImpl ns = StepBuilder
                    .initFullyConnected(hiddenLayersSizes.get(i), in.getNeuronsWithBias(), Option.of(this.activationFunction))
                    .setBias(this.bias)
                    .setMean(this.mean)
                    .setRange(this.range)
                    .setMomentum(this.momentum)
                    .setLearningRate(this.learningRate)
                    .setNamePrefix(this.name + " ")
                    .setRandom(this.random)
                    .buildWithConnections();

            stepHiddenList = stepHiddenList.append(ns);
            in = ns;
        }

        StepImpl outputStep = StepBuilder
                .initFullyConnected(
                        outputSize,
                        stepHiddenList.isEmpty() ? stepInput.getNeuronsWithBias() : stepHiddenList.last().getNeuronsWithBias(),
                        Option.none()
                )
                .setBias(this.bias)
                .setMean(this.mean)
                .setRange(this.range)
                .setMomentum(this.momentum)
                .setLearningRate(this.learningRate)
                .setNamePrefix(this.name + " ")
                .setRandom(this.random)
                .buildWithConnections();

        return new Network(
                stepInput,
                stepHiddenList,
                outputStep,
                validationFunction,
                this.name,
                this.outputResolver
        );
    }


    public NetworkBuilder setBias(Double bias) {
        this.bias = bias;
        return this;
    }

    public NetworkBuilder setMean(Double mean) {
        this.mean = mean;
        return this;
    }

    public NetworkBuilder setRange(Double range) {
        this.range = range;
        return this;
    }

    public NetworkBuilder setMomentum(Double momentum) {
        this.momentum = momentum;
        return this;
    }

    public NetworkBuilder setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NetworkBuilder setRandom(Random random) {
        this.random = random;
        return this;
    }

    public NetworkBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public NetworkBuilder setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
        return this;
    }

    public NetworkBuilder setOutputResolver(OutputResolver outputResolver) {
        this.outputResolver = outputResolver;
        return this;
    }
}
