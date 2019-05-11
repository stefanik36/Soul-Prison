package com.stefanik36.soul_prison.network_components.neuron;


import com.stefanik36.soul_prison.util.ActivationFunction;
import io.vavr.collection.List;
import io.vavr.control.Option;

public class NeuronNode implements Neuron {

    private String name;
    private Double learningRate;
    private Option<ActivationFunction> oActivationFunction;

    private List<Synapse> connectionList;

    private Double lastOutput;
    private Double error;
    private Double momentum;
    private boolean frozenWeights;

    public NeuronNode(
            String name,
            List<Synapse> connectionList,
            Option<ActivationFunction> oActivationFunction,
            Double learningRate,
            Double momentum
    ) {
        this.name = name;
        this.oActivationFunction = oActivationFunction;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.connectionList = connectionList;
        this.error = null;
        this.frozenWeights = false;
    }

    @Override
    public void propagate() {
        Double sum = connectionList
                .map(c -> c.getWeight() * c.getNeuron().getLastOutput())
                .sum()
                .doubleValue();
        this.lastOutput = oActivationFunction
                .map(af -> af.apply(sum))
                .getOrElse(sum);
    }

    @Override
    public Double getLastOutput() {
        return lastOutput;
    }

    @Override
    public Double getError() {
        return error;
    }

    @Override
    public void updateWeights() {
        for (Synapse connection : connectionList.filter(c -> !c.getNeuron().isFrozenWeights())) {
            connection.setWeight(connection.getWeight() + connection.getLastWeight() * momentum + connection.getNewWeight());
            connection.setLastWeight(connection.getNewWeight());
        }
    }

    @Override
    public void setError(Double error) {
        this.error = error;
    }

    @Override
    public void backPropagate() {
        connectionList.filter(c -> !c.getNeuron().isFrozenWeights())
                .forEach(c -> c.setNewWeight(error
                        * this.oActivationFunction.map($ -> lastOutput * (1 - lastOutput)).getOrElse(1.0)
                        * c.getNeuron().getLastOutput()
                        * learningRate
                ));
    }

    @Override
    public void computePrevStepError() {
        connectionList.filter(c -> (c.getNeuron() instanceof NeuronNode))
                .forEach(c -> c.getNeuron().setError(
                        c.getNeuron().getError()
                                + this.error
                                * this.oActivationFunction.map($ -> lastOutput * (1 - lastOutput)).getOrElse(1.0)
                                * c.getWeight()
                ));
    }

    public void removeConnection(Neuron neuron) {
        connectionList = connectionList.reject(c -> c.getNeuron().equals(neuron));
    }

    public void addConnection(Synapse synapse) {
        connectionList = connectionList.append(synapse);//TODO TEST
    }

    public void setSynapseList(List<Synapse> connectionList) {
        this.connectionList = connectionList;
    }

    public void setConnectionsErrorTo(double value) {
        connectionList
//                .filter(c -> c.getNeuron().getError() == null || Double.isNaN(c.getNeuron().getError()))
                .forEach(c -> c.getNeuron().setError(value));
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
