package com.stefanik36.soul_prison.network_components.layer;

import com.stefanik36.soul_prison.network_components.neuron.Neuron;
import io.vavr.collection.List;

public interface Step {
    List<Neuron> getNeuronsWithBias();

    String getName();

    default void setFrozenWeights(boolean frozenWeights) {
        getNeuronsWithBias().forEach(n -> n.setFrozenWeights(frozenWeights));
    }

    void setName(String name);
}
