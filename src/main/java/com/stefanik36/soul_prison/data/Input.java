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

    @Deprecated
    public Input getNormalized() {
        Double xMin = values.min().getOrElseThrow(() -> new RuntimeException("No minimal value."));
        Double xMax = values.max().getOrElseThrow(() -> new RuntimeException("No maximal value."));
        return doNormalize(xMin, xMax);
    }

    public Input getNormalizedByAll(List<DataResultTuple> iData) {
        Double xMin = iData.flatMap(DataResultTuple::getData).min().getOrElseThrow(() -> new RuntimeException("No minimal value."));
        Double xMax = iData.flatMap(DataResultTuple::getData).max().getOrElseThrow(() -> new RuntimeException("No maximal value."));
        return doNormalize(xMin, xMax);
    }

    public Input doNormalize(Double xMin, Double xMax) {
        if (xMax.equals(xMin)) {
            return new Input(this.getValues());
        }
        return new Input(values.map(x -> (x - xMin) / (xMax - xMin)));
    }


}
