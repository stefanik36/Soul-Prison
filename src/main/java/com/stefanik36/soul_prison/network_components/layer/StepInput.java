package com.stefanik36.soul_prison.network_components.layer;

import com.stefanik36.soul_prison.data.Input;
import com.stefanik36.soul_prison.network_components.neuron.Neuron;
import com.stefanik36.soul_prison.network_components.neuron.NeuronBias;
import com.stefanik36.soul_prison.network_components.neuron.NeuronInput;
import io.vavr.collection.List;

public class StepInput implements Step {
    private List<NeuronInput> neuronInputList;
    private NeuronBias neuronBias;
    private String name;


    public StepInput(List<NeuronInput> neuronInputList, NeuronBias neuronBias, String name) {
        this.neuronInputList = neuronInputList;
        this.neuronBias = neuronBias;
        this.name = name;
    }

    public StepInput(List<NeuronInput> neuronInputList, NeuronBias neuronBias) {
        this(neuronInputList, neuronBias, "");
    }

    public void setInput(Input input) {
        neuronInputList
                .zip(input.getValues())
                .forEach(t -> t._1().setValue(t._2().doubleValue()));
    }

    public List<Neuron> getNeuronsWithBias() {
        return List.of((Neuron) neuronBias).appendAll(neuronInputList);
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public void setName(String name) {
        this.name = name;
    }
}
