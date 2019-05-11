package com.stefanik36.soul_prison.builder;

import com.stefanik36.soul_prison.network_components.layer.Step;
import com.stefanik36.soul_prison.network_components.layer.StepImpl;
import com.stefanik36.soul_prison.network_components.layer.StepInput;
import com.stefanik36.soul_prison.network_components.neuron.*;
import com.stefanik36.soul_prison.util.ActivationFunction;
import io.vavr.collection.List;
import io.vavr.control.Option;

import java.util.Random;
import java.util.function.Supplier;

public class StepBuilder {
    private static final Double MEAN = 0.0;
    private static final Double RANGE = 1.0;
    private static final Double MOMENTUM = 0.4;
    private static final Double LEARNING_RATE = 0.3;
    private static final String NAME_PREFIX = "";
    private static final Double BIAS = 1.0;

    private static int build = 0;

    private Double bias;
    private Double mean;
    private Double range;
    private Double momentum;
    private Double learningRate;
    private String namePrefix;

    private Random random;
    private String name;

    private BuilderType builderType;
    private int inputSize;
    private List<Neuron> prevNeurons;
    private Option<ActivationFunction> activationFunction;

    private StepImpl toUpdate;


    private StepBuilder(BuilderType builderType, Integer inputSize, List<Neuron> prevNeurons, Option<ActivationFunction> activationFunction, StepImpl toUpdate) {
        this.bias = BIAS;
        this.mean = MEAN;
        this.range = RANGE;
        this.momentum = MOMENTUM;
        this.learningRate = LEARNING_RATE;
        this.namePrefix = NAME_PREFIX;

        this.name = "s_" + build;
        this.random = new Random();

        this.builderType = builderType;
        this.inputSize = inputSize;
        this.prevNeurons = prevNeurons;
        this.activationFunction = activationFunction;

        this.toUpdate = toUpdate;
    }

    private StepBuilder(BuilderType builderType, Integer inputSize) {
        this(builderType, inputSize, null, null, null);
    }

    private StepBuilder(BuilderType builderType, StepImpl toUpdate, List<Neuron> prevNeurons) {
        this(builderType, 0, prevNeurons, null, toUpdate);
    }


    /*
     * Initialize methods
     */

    public static StepBuilder initInput(int inputSize) {
        return new StepBuilder(BuilderType.INPUT, inputSize);
    }

    public static StepBuilder initFullyConnected(Integer size, List<Neuron> prevNeurons, Option<ActivationFunction> activationFunction) {
        return new StepBuilder(BuilderType.FULLY_CONNECTED, size, prevNeurons, activationFunction, null);
    }

    public static StepBuilder updatePrevNeurons(StepImpl toUpdate, List<Neuron> prevNeurons) {
        return new StepBuilder(BuilderType.UPDATE, toUpdate, prevNeurons);
    }


    /*
     * Build methods
     */

    public StepInput buildInput() {
        return (StepInput) doBuild();
    }

    public StepImpl buildWithConnections() {
        return (StepImpl) doBuild();
    }

    private Step doBuild() {
        Step result;
        switch (builderType) {
            case INPUT:
                result = buildInputStep();
                break;
            case FULLY_CONNECTED:
                result = buildStepImpl();
                break;
            case UPDATE:
                result = update();
                break;
            default:
                throw new UnsupportedOperationException();
        }
        build++;
        return result;
    }

    private StepInput buildInputStep() {
        String fullName = this.namePrefix + this.name;
        return new StepInput(List.fill(
                inputSize,
                () -> NeuronBuilder.initInput()
                        .setNamePrefix(fullName + " ")
                        .buildInput()),
                NeuronBuilder.initBias(bias)
                        .setNamePrefix(fullName + " ")
                        .buildBias(),
                fullName
        );
    }

    private StepImpl buildStepImpl() {
        String fullName = this.namePrefix + this.name;
        return new StepImpl(
                List.fill(
                        inputSize,
                        () -> NeuronBuilder
                                .init(prevNeurons, this.activationFunction)
                                .setMean(this.mean)
                                .setRange(this.range)
                                .setMomentum(this.momentum)
                                .setLearningRate(this.learningRate)
                                .setNamePrefix(fullName + " ")
                                .setRandom(this.random)
                                .build()
                ),
                NeuronBuilder.initBias(bias)
                        .setNamePrefix(fullName + " ")
                        .buildBias(),
                fullName
        );
    }

    private StepImpl update() {
        Supplier<Double> weightSupplier = () -> ((this.random.nextDouble() - 0.5) * this.range * 2) + this.mean;
        toUpdate.getNeurons()
                .filter(n -> n instanceof NeuronNode)
                .map(n -> (NeuronNode) n)
                .forEach(n -> n.setSynapseList(this.prevNeurons.map(pn -> new Synapse(pn, weightSupplier.get()))));
        return toUpdate;
    }


    enum BuilderType {
        INPUT, FULLY_CONNECTED, UPDATE;
    }

    public StepBuilder setBias(Double bias) {
        this.bias = bias;
        return this;
    }

    public StepBuilder setMean(Double mean) {
        this.mean = mean;
        return this;
    }

    public StepBuilder setRange(Double range) {
        this.range = range;
        return this;
    }

    public StepBuilder setMomentum(Double momentum) {
        this.momentum = momentum;
        return this;
    }

    public StepBuilder setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public StepBuilder setNamePrefix(String namePrefix) {
        this.namePrefix = namePrefix;
        return this;
    }

    public StepBuilder setRandom(Random random) {
        this.random = random;
        return this;
    }

    public StepBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public String getName() {
        return name;
    }
}