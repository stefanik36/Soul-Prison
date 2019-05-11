package com.stefanik36.soul_prison.network_components.neuron;

public class NeuronInput implements Neuron {

    private Double value;
    private Boolean frozenWeights;
    private String name;

    public NeuronInput(String name) {
        this.name = name;
        frozenWeights = false;
        value = null;
    }

    @Override
    public void setError(Double error) {

    }

    @Override
    public Double getLastOutput() {
        return value;
    }

    @Override
    public Double getError() {
        return null;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    @Override
    public void propagate() {
        //Nothing
        throw new RuntimeException("Should not be used.");
    }

    @Override
    public void backPropagate() {
        //Nothing
        throw new RuntimeException("Should not be used.");
    }

    @Override
    public void computePrevStepError() {
        //Nothing
        throw new RuntimeException("Should not be used.");
    }

    @Override
    public void updateWeights() {
        //Nothing
        throw new RuntimeException("Should not be used.");
    }

    @Override
    public void setFrozenWeights(boolean frozenWeights) {
        this.frozenWeights = frozenWeights;
    }

    @Override
    public boolean isFrozenWeights() {
        return frozenWeights;
    }

    @Override
    public String getName() {
        return this.name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }

}
