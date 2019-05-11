package com.stefanik36.soul_prison.network_components.layer;

import com.stefanik36.soul_prison.data.Input;
import com.stefanik36.soul_prison.network_components.neuron.Neuron;
import com.stefanik36.soul_prison.network_components.neuron.NeuronBias;
import com.stefanik36.soul_prison.network_components.neuron.NeuronNode;
import io.vavr.collection.List;
import io.vavr.control.Option;

import java.util.Objects;

public class StepImpl implements Step {

    private List<Neuron> neurons;
    private NeuronBias bias;
    private String name;

    public StepImpl(List<Neuron> neurons, NeuronBias bias, String name) {
        this.neurons = neurons;
        this.bias = bias;
        this.name = name;
    }

    public StepImpl(List<Neuron> neurons, NeuronBias bias) {
        this.neurons = neurons;
        this.bias = bias;
    }


    public void propagate() {
        neurons.forEach(Neuron::propagate);
    }


    public void backPropagate(Option<StepImpl> prev) {
        if (!prev.isEmpty()) {
            computePrevStepError(prev.get());
        }
        neurons.forEach(Neuron::backPropagate);
    }

    private void computePrevStepError(StepImpl prevStep) {
//        prevStep.getNeuronsWithBias().forEach(n -> n.setError(0.0));
        this.neurons.forEach(n -> ((NeuronNode) n).setConnectionsErrorTo(0.0));
        this.neurons.forEach(Neuron::computePrevStepError);
    }

    @Override
    public List<Neuron> getNeuronsWithBias() {
        return List.of((Neuron) bias).appendAll(neurons);
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public void updateWeights() {
        neurons.forEach(Neuron::updateWeights);
    }

    public void computeError(Input result) {
        if (result.getValues().size() != neurons.size()) {
            throw new RuntimeException("Output size (" + result.getValues().size() + ") do not equal output layer size(" + neurons.size() + ").");
        }
        neurons.zip(result.getValues()).forEach(t -> t._1().setError(t._2() - t._1().getLastOutput()));
    }

    public List<Double> getResults() {
        return neurons.map(Neuron::getLastOutput);
    }

    @Override
    public String getName() {
        return name;
    }

    public Double getSquareError() {
        return neurons.map(Neuron::getError)
                .filter(Objects::nonNull)
                .map(v -> v * v)
                .sum()
                .doubleValue();
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }
}
