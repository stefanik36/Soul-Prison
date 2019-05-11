package com.stefanik36.soul_prison.data;


import io.vavr.collection.List;

public class Input {


    private List<Double> values;

    public Input(List<Double> values) {
        this.values = values;
    }


    public List<Double> getValues() {
        return values;
    }

    public Input getNormalized() {
        Double xMin = values.min().getOrElseThrow(() -> new RuntimeException("No minimal value."));
        Double xMax = values.max().getOrElseThrow(() -> new RuntimeException("No maximal value."));
        if (xMax.equals(xMin)) {
            return new Input(this.getValues());
        }
        return new Input(values.map(x -> (x - xMin) / (xMax - xMin)));
    }
}
