package com.stefanik36.soul_prison.data;

public class InOut {

    private Input input;
    private Input output;

    public InOut(Input input, Input output) {
        this.input = input;
        this.output = output;
    }

    public Input getInput() {
        return input;
    }

    public Input getOutput() {
        return output;
    }
}