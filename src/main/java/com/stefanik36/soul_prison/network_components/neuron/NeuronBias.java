package com.stefanik36.soul_prison.network_components.neuron;

public class NeuronBias implements Neuron {

    private String name;
    private Double error;
    private Double biasValue;
    private boolean frozenWeights;

    public NeuronBias(double biasValue, String name) {
        this.error = null;
        this.biasValue = biasValue;
        this.frozenWeights = false;
        this.name = name;
    }


    public Double getBiasValue() {
        return biasValue;
    }

    public void setBiasValue(Double biasValue) {
        this.biasValue = biasValue;
    }

    @Override
    public void setError(Double error) {
        this.error = error;
    }

    @Override
    public Double getLastOutput() {
        return biasValue;
    }

    @Override
    public Double getError() {
        return error;
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
    public void setName(String name) {
        this.name = name;
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
        return name;
    }
}
