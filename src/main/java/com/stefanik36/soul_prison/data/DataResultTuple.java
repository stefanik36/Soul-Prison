package com.stefanik36.soul_prison.data;

import io.vavr.collection.List;

public class DataResultTuple {

    private Double result;
    private List<Double> data;

    public DataResultTuple(Double result, List<Double> data) {
        this.result = result;
        this.data = data;
    }

    public List<Double> getClassificationResult(List<Double> possibleValues) {
        return possibleValues.map(pv -> result.equals(pv) ? 1.0 : 0.0);
    }

    public Double getResult() {
        return result;
    }

    public void setResult(Double result) {
        this.result = result;
    }

    public List<Double> getData() {
        return data;
    }

    public void setData(List<Double> data) {
        this.data = data;
    }
}
