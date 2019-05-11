package com.stefanik36.soul_prison.network_components.neuron;

public class Synapse {
    private Neuron neuron;
    private Double lastWeight;
    private Double weight;
    private Double newWeight;

    public Synapse(Neuron neuron, Double weight) {
        this.neuron = neuron;
        this.weight = weight;
        this.lastWeight = 0.0;
    }

    public Neuron getNeuron() {
        return neuron;
    }

    public Double getWeight() {
        return weight;
    }

    public Double getNewWeight() {
        return newWeight;
    }

    public void setNewWeight(Double newWeight) {
        this.newWeight = newWeight;
    }

    public Double getLastWeight() {
        return lastWeight;
    }

    public void setLastWeight(Double lastWeight) {
        this.lastWeight = lastWeight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }
}
