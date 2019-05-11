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

public class NeuronBuilder {
    private static final Double MEAN = 0.0;
    private static final Double RANGE = 1.0;
    private static final Double MOMENTUM = 0.4;
    private static final Double LEARNING_RATE = 0.3;
    private static final String NAME_PREFIX = "";

    private static int build = 0;
    private BuilderType builderType;


    enum BuilderType {
        INPUT, NODE, BIAS
    }

    private Double mean;
    private Double range;
    private Double momentum;
    private Double learningRate;
    private String namePrefix;

    private Random random;
    private String name;

    /**
     * mandatory
     */
    private List<Neuron> neurons;
    private Option<ActivationFunction> activationFunction;
    private Double bias;


    public NeuronBuilder(BuilderType builderType, List<Neuron> neurons, Option<ActivationFunction> activationFunction, Double bias) {
        this.builderType = builderType;
        this.mean = MEAN;
        this.range = RANGE;
        this.momentum = MOMENTUM;
        this.learningRate = LEARNING_RATE;
        this.namePrefix = NAME_PREFIX;

        this.name = "n_" + build;
        this.random = new Random();

        this.neurons = neurons;
        this.activationFunction = activationFunction;
        this.bias = bias;
    }


    public static NeuronBuilder init(List<Neuron> neurons, Option<ActivationFunction> activationFunction) {
        return new NeuronBuilder(BuilderType.NODE, neurons, activationFunction, null);
    }

    public static NeuronBuilder initInput() {
        return new NeuronBuilder(BuilderType.INPUT, null, null, null);
    }

    public static NeuronBuilder initBias(Double bias) {
        return new NeuronBuilder(BuilderType.BIAS, null, null, bias);
    }


    public NeuronInput buildInput() {
        return (NeuronInput) build();
    }

    public NeuronBias buildBias() {
        return (NeuronBias) build();
    }

    public Neuron build() {
        Neuron result;
        switch (builderType) {
            case NODE:
                result = buildNeuronNode();
                break;
            case INPUT:
                result = buildNeuronInput();
                break;
            case BIAS:
                result = buildNeuronBias();
                break;
            default:
                throw new UnsupportedOperationException();
        }
        build++;
        return result;
    }

    private NeuronBias buildNeuronBias() {
        String fullName = this.namePrefix + this.name;
        return new NeuronBias(this.bias, fullName);
    }

    private NeuronNode buildNeuronNode() {
        Supplier<Double> weightSupplier = () -> ((this.random.nextDouble() - 0.5) * this.range * 2) + this.mean;
        String fullName = this.namePrefix + this.name;
        return new NeuronNode(
                fullName,
                this.neurons.map(n -> new Synapse(n, weightSupplier.get())),
                this.activationFunction,
                this.learningRate,
                this.momentum
        );
    }

    private Neuron buildNeuronInput() {
        String fullName = this.namePrefix + this.name;
        return new NeuronInput(fullName);
    }


    public NeuronBuilder setMean(Double mean) {
        this.mean = mean;
        return this;
    }

    public NeuronBuilder setRange(Double range) {
        this.range = range;
        return this;
    }

    public NeuronBuilder setMomentum(Double momentum) {
        this.momentum = momentum;
        return this;
    }

    public NeuronBuilder setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public NeuronBuilder setNamePrefix(String namePrefix) {
        this.namePrefix = namePrefix;
        return this;
    }

    public NeuronBuilder setRandom(Random random) {
        this.random = random;
        return this;
    }

    public NeuronBuilder setName(String name) {
        this.name = name;
        return this;
    }

    public String getName() {
        return name;
    }
}