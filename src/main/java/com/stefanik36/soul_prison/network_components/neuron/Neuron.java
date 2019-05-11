package com.stefanik36.soul_prison.network_components.neuron;

public interface Neuron {

    void setError(Double error);

    Double getLastOutput();

    Double getError();

    void propagate();

    void backPropagate();

    void computePrevStepError();

    void updateWeights();

    void setFrozenWeights(boolean frozenWeights);

    boolean isFrozenWeights();

    String getName();

    void setName(String name);
}
